#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Device function for partial sum computations
__device__ void computePartialSums(const float* __restrict__ input,
                                   int c, int N, int C, int H, int W,
                                   int tid, int stride,
                                   float &partialSum, float &partialSumSq) {
    int numElements = N * H * W;
    partialSum = 0.0f;
    partialSumSq = 0.0f;
    for (int i = tid; i < numElements; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        partialSum += val;
        partialSumSq += val * val;
    }
}

// Device function for block-level reduction
__device__ inline void blockReduce(float* sum, float* sumSq, int blockDim) {
    int tid = threadIdx.x;
    for (int s = blockDim / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sum[tid] += sum[tid + s];
            sumSq[tid] += sumSq[tid + s];
        }
        __syncthreads();
    }
}

// Device function for normalization
__device__ inline float normalizeValue(float val, float mean, float invStd, float w, float b) {
    return (val - mean) * invStd * w + b;
}

__global__ void optimized_batch_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    bool training,
    float momentum,
    float eps,
    float* __restrict__ output,
    int N, int C, int H, int W) {

    int c = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int numElements = N * H * W;

    extern __shared__ float smem[];
    float* sum_shared = smem;            
    float* sumSq_shared = &smem[blockDim.x];

    float partialSum, partialSumSq;
    computePartialSums(input, c, N, C, H, W, tid, stride, partialSum, partialSumSq);
    sum_shared[tid] = partialSum;
    sumSq_shared[tid] = partialSumSq;
    __syncthreads();

    blockReduce(sum_shared, sumSq_shared, blockDim.x);

    float mean, var;
    if (tid == 0) {
        float totalSum = sum_shared[0];
        float totalSumSq = sumSq_shared[0];
        mean = totalSum / numElements;
        var = totalSumSq / numElements - mean * mean;
        if (training) {
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
        smem[0] = mean;
        smem[1] = var;
    }
    __syncthreads();
    mean = smem[0];
    var = smem[1];

    float invStd = rsqrtf(var + eps);
    float channelWeight = weight[c];
    float channelBias = bias[c];

    for (int i = tid; i < numElements; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w_idx = hw % W;
        int idx = ((n * C + c) * H + h) * W + w_idx;
        float val = input[idx];
        output[idx] = normalizeValue(val, mean, invStd, channelWeight, channelBias);
    }
}

torch::Tensor optimized_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {

    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);
    CHECK_CUDA(running_mean);
    CHECK_CUDA(running_var);

    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(bias);
    CHECK_CONTIGUOUS(running_mean);
    CHECK_CONTIGUOUS(running_var);

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty_like(input);

    int threads = 256;  // balanced threads and blocks for CUDA capability
    size_t shared_mem = 2 * threads * sizeof(float);

    optimized_batch_norm_kernel<<<C, threads, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        training,
        momentum,
        eps,
        output.data_ptr<float>(),
        N, C, H, W
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_forward_cuda, "Optimized BatchNorm forward (CUDA)");
}
