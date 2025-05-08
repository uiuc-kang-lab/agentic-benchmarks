#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Two-phase block reduction combining warp shuffle and shared memory
__device__ void hybridBlockReduce(float &sum, float &sumSq) {
    // First phase: warp-level reduction
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    sum = warpReduceSum(sum);
    sumSq = warpReduceSum(sumSq);
    
    __shared__ float warpSum[32];    // Max 32 warps per block
    __shared__ float warpSumSq[32];
    
    // Store warp results
    if (lane == 0) {
        warpSum[wid] = sum;
        warpSumSq[wid] = sumSq;
    }
    __syncthreads();
    
    // Second phase: final reduction by first warp
    if (threadIdx.x < warpSize) {
        sum = (threadIdx.x < (blockDim.x / warpSize)) ? warpSum[threadIdx.x] : 0;
        sumSq = (threadIdx.x < (blockDim.x / warpSize)) ? warpSumSq[threadIdx.x] : 0;
        sum = warpReduceSum(sum);
        sumSq = warpReduceSum(sumSq);
    }
}

__global__ void optimized_batchnorm_kernel(
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
    
    // Coalesced memory access pattern
    float partial_sum = 0.0f;
    float partial_sum_sq = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < numElements; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        partial_sum += val;
        partial_sum_sq += val * val;
    }
    
    // Efficient hybrid reduction
    hybridBlockReduce(partial_sum, partial_sum_sq);
    
    __shared__ float mean_var[2];
    if (tid == 0) {
        float mean = partial_sum / numElements;
        float var = (partial_sum_sq / numElements) - (mean * mean);
        
        if (training) {
            atomicExch(&running_mean[c], (1 - momentum) * running_mean[c] + momentum * mean);
            atomicExch(&running_var[c], (1 - momentum) * running_var[c] + momentum * var);
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
        mean_var[0] = mean;
        mean_var[1] = var;
    }
    __syncthreads();
    
    float mean = mean_var[0];
    float var = mean_var[1];
    float inv_std = rsqrtf(var + eps);
    float w = weight[c];
    float b = bias[c];
    
    // Vectorized output computation
    #pragma unroll 4
    for (int i = tid; i < numElements; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w_idx = hw % W;
        int idx = ((n * C + c) * H + h) * W + w_idx;
        output[idx] = (input[idx] - mean) * inv_std * w + b;
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
    
    int threads = 256;  // Optimized thread count
    
    optimized_batchnorm_kernel<<<C, threads>>>(
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