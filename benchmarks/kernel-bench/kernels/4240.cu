#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define WARP_SIZE 32

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void minimal_sync_batchnorm_kernel(
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

    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int warpsPerBlock = blockDim.x / WARP_SIZE;
    const int numElements = N * H * W;

    __shared__ float warpSums[32];    // For mean calculation
    __shared__ float warpSumSqs[32];  // For variance calculation
    __shared__ float channelMean;
    __shared__ float channelInvStd;

    // Phase 1: Compute partial sums within each thread
    float sum = 0.0f;
    float sumSq = 0.0f;

    // Stride across elements, maintaining coalesced access pattern
    for (int i = tid; i < numElements; i += blockDim.x) {
        const int n = i / (H * W);
        const int hw = i % (H * W);
        const int h = hw / W;
        const int w = hw % W;
        const int idx = ((n * C + c) * H + h) * W + w;
        const float val = input[idx];
        sum += val;
        sumSq += val * val;
    }

    // Warp-level reduction
    sum = warpReduceSum(sum);
    sumSq = warpReduceSum(sumSq);

    // Store warp results in shared memory
    if (lane == 0) {
        warpSums[warpId] = sum;
        warpSumSqs[warpId] = sumSq;
    }

    // Single sync point to ensure warp results are visible
    __syncthreads();

    // Final reduction by first warp
    if (warpId == 0) {
        sum = (lane < warpsPerBlock) ? warpSums[lane] : 0.0f;
        sumSq = (lane < warpsPerBlock) ? warpSumSqs[lane] : 0.0f;
        
        sum = warpReduceSum(sum);
        sumSq = warpReduceSum(sumSq);

        if (lane == 0) {
            float mean = sum / numElements;
            float var = (sumSq / numElements) - (mean * mean);
            
            if (training) {
                running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
                running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
            } else {
                mean = running_mean[c];
                var = running_var[c];
            }

            channelMean = mean;
            channelInvStd = rsqrtf(var + eps);
        }
    }

    // Single sync point before normalization phase
    __syncthreads();

    // Load channel-specific parameters once
    const float w = weight[c];
    const float b = bias[c];
    const float mean = channelMean;
    const float invStd = channelInvStd;

    // Phase 2: Normalize and write output with coalesced access
    #pragma unroll 4
    for (int i = tid; i < numElements; i += blockDim.x) {
        const int n = i / (H * W);
        const int hw = i % (H * W);
        const int h = hw / W;
        const int w_idx = hw % W;
        const int idx = ((n * C + c) * H + h) * W + w_idx;
        const float val = input[idx];
        output[idx] = (val - mean) * invStd * w + b;
    }
}

torch::Tensor forward_cuda(
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

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty_like(input);
    
    const int threads = 256;
    
    minimal_sync_batchnorm_kernel<<<C, threads>>>(
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
    m.def("forward", &forward_cuda, "BatchNorm forward (CUDA)");
}