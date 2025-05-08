#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Two-phase block reduction combining warp shuffle and shared memory
__device__ void hybridBlockReduce(float &sum, float &sum_sq) {
    // First do warp-level reduction
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    // Warp reduction first
    sum = warpReduceSum(sum);
    sum_sq = warpReduceSum(sum_sq);
    
    __shared__ float warp_sums[32][2];  // Assuming max 32 warps
    
    // Write reduced warp results to shared memory
    if (lane == 0) {
        warp_sums[wid][0] = sum;
        warp_sums[wid][1] = sum_sq;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (wid == 0) {
        sum = (lane < (blockDim.x + warpSize - 1)/warpSize) ? warp_sums[lane][0] : 0;
        sum_sq = (lane < (blockDim.x + warpSize - 1)/warpSize) ? warp_sums[lane][1] : 0;
        
        sum = warpReduceSum(sum);
        sum_sq = warpReduceSum(sum_sq);
    }
    __syncthreads();
}

__global__ void hybrid_batch_norm_kernel(
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
    const int num_elements = N * H * W;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    __shared__ float mean_var[2];  // [mean, var]
    
    if (training) {
        // Phase 1: Compute statistics using hybrid reduction
        float sum = 0.0f, sum_sq = 0.0f;
        
        // Coalesced memory access pattern
        for (int i = tid; i < num_elements; i += stride) {
            const int n = i / (H * W);
            const int hw = i % (H * W);
            const int idx = ((n * C + c) * H + hw/W) * W + hw%W;
            const float val = input[idx];
            sum += val;
            sum_sq += val * val;
        }
        
        // Hybrid reduction combining warp shuffle and shared memory
        hybridBlockReduce(sum, sum_sq);
        
        if (tid == 0) {
            float mean = sum / num_elements;
            float var = (sum_sq / num_elements) - (mean * mean);
            
            // Update running stats
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
            
            mean_var[0] = mean;
            mean_var[1] = var;
        }
        __syncthreads();
    } else {
        if (tid == 0) {
            mean_var[0] = running_mean[c];
            mean_var[1] = running_var[c];
        }
        __syncthreads();
    }
    
    // Phase 2: Normalize with vectorized memory access
    const float mean = mean_var[0];
    const float inv_std = rsqrtf(mean_var[1] + eps);
    const float w = weight[c];
    const float b = bias[c];
    
    // Use vectorized loads/stores where possible
    #pragma unroll 4
    for (int i = tid; i < num_elements; i += stride) {
        const int n = i / (H * W);
        const int hw = i % (H * W);
        const int idx = ((n * C + c) * H + hw/W) * W + hw%W;
        const float val = input[idx];
        output[idx] = (val - mean) * inv_std * w + b;
    }
}

torch::Tensor hybrid_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {
    
    CHECK_CUDA(input); CHECK_CUDA(weight); CHECK_CUDA(bias);
    CHECK_CUDA(running_mean); CHECK_CUDA(running_var);
    CHECK_CONTIGUOUS(input); CHECK_CONTIGUOUS(weight); CHECK_CONTIGUOUS(bias);
    CHECK_CONTIGUOUS(running_mean); CHECK_CONTIGUOUS(running_var);
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    hybrid_batch_norm_kernel<<<C, threads>>>(
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
    m.def("forward", &hybrid_forward_cuda, "Hybrid BatchNorm forward (CUDA)");
}