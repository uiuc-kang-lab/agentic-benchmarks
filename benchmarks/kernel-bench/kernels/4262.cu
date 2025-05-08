#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void partial_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ partial_sums,
    float* __restrict__ partial_sums_sq,
    int N, int C, int H, int W,
    int blocksPerChannel) {

    int c = blockIdx.x;
    int blockId = blockIdx.y;
    int tid = threadIdx.x;
    int numElements = N * H * W;
    int elementsPerBlock = (numElements + blocksPerChannel - 1) / blocksPerChannel;
    int start = blockId * elementsPerBlock;
    int end = min(start + elementsPerBlock, numElements);

    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = start + tid; i < end; i += blockDim.x) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        sum += val;
        sum_sq += val * val;
    }

    sum = warpReduceSum(sum);
    sum_sq = warpReduceSum(sum_sq);

    __shared__ float s_sum[32], s_sumsq[32];
    int warpId = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;

    if (lane == 0) {
        s_sum[warpId] = sum;
        s_sumsq[warpId] = sum_sq;
    }
    __syncthreads();

    if (warpId == 0) {
        sum = lane < (blockDim.x + warpSize - 1)/warpSize ? s_sum[lane] : 0;
        sum_sq = lane < (blockDim.x + warpSize - 1)/warpSize ? s_sumsq[lane] : 0;
        sum = warpReduceSum(sum);
        sum_sq = warpReduceSum(sum_sq);
        
        if (lane == 0) {
            int idx = c * blocksPerChannel + blockId;
            partial_sums[idx] = sum;
            partial_sums_sq[idx] = sum_sq;
        }
    }
}

__global__ void finalize_normalize_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    float* __restrict__ output,
    const float* __restrict__ partial_sums,
    const float* __restrict__ partial_sums_sq,
    int blocksPerChannel,
    bool training,
    float momentum,
    float eps,
    int N, int C, int H, int W) {

    __shared__ float s_mean, s_invStd, s_weight, s_bias;
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int numElements = N * H * W;

    float total_sum = 0.0f, total_sum_sq = 0.0f;
    for (int i = tid; i < blocksPerChannel; i += blockDim.x) {
        total_sum += partial_sums[c * blocksPerChannel + i];
        total_sum_sq += partial_sums_sq[c * blocksPerChannel + i];
    }

    total_sum = warpReduceSum(total_sum);
    total_sum_sq = warpReduceSum(total_sum_sq);

    __shared__ float s_psum[32], s_psumsq[32];
    int warpId = tid / warpSize;
    int lane = tid % warpSize;

    if (lane == 0) {
        s_psum[warpId] = total_sum;
        s_psumsq[warpId] = total_sum_sq;
    }
    __syncthreads();

    if (warpId == 0) {
        total_sum = lane < (blockDim.x + warpSize - 1)/warpSize ? s_psum[lane] : 0;
        total_sum_sq = lane < (blockDim.x + warpSize - 1)/warpSize ? s_psumsq[lane] : 0;
        total_sum = warpReduceSum(total_sum);
        total_sum_sq = warpReduceSum(total_sum_sq);

        if (lane == 0) {
            float mean = total_sum / numElements;
            float var = total_sum_sq / numElements - mean*mean;
            
            if (training) {
                running_mean[c] = (1 - momentum)*running_mean[c] + momentum*mean;
                running_var[c] = (1 - momentum)*running_var[c] + momentum*var;
            } else {
                mean = running_mean[c];
                var = running_var[c];
            }

            s_mean = mean;
            s_invStd = rsqrtf(var + eps);
            s_weight = weight[c];
            s_bias = bias[c];
        }
    }
    __syncthreads();

    for (int i = tid; i < numElements; i += blockDim.x) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        output[idx] = (val - s_mean) * s_invStd * s_weight + s_bias;
    }
}

torch::Tensor combined_batchnorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {

    CHECK_CUDA(input); CHECK_CONTIGUOUS(input);
    CHECK_CUDA(weight); CHECK_CONTIGUOUS(weight);
    CHECK_CUDA(bias); CHECK_CONTIGUOUS(bias);
    CHECK_CUDA(running_mean); CHECK_CONTIGUOUS(running_mean);
    CHECK_CUDA(running_var); CHECK_CONTIGUOUS(running_var);

    int N = input.size(0), C = input.size(1);
    int H = input.size(2), W = input.size(3);
    int numElements = N * H * W;
    auto output = torch::empty_like(input);

    // Configure partial reduction
    int threads = 256;
    int blocksPerChannel = (numElements + threads - 1) / threads;
    blocksPerChannel = min(blocksPerChannel, 32);
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto partial_sums = torch::zeros({C * blocksPerChannel}, options);
    auto partial_sums_sq = torch::zeros({C * blocksPerChannel}, options);

    // Phase 1: Partial reductions
    dim3 grid_reduce(C, blocksPerChannel);
    partial_reduce_kernel<<<grid_reduce, threads>>>(
        input.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        partial_sums_sq.data_ptr<float>(),
        N, C, H, W,
        blocksPerChannel
    );

    // Phase 2: Final reduction + normalization
    dim3 grid_finalize(C);
    finalize_normalize_kernel<<<grid_finalize, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        partial_sums_sq.data_ptr<float>(),
        blocksPerChannel,
        training,
        momentum,
        eps,
        N, C, H, W
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &combined_batchnorm_forward, "Combined Batchnorm CUDA");
}
