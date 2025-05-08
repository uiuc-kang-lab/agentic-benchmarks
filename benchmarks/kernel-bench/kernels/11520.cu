#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel for KL divergence calculation
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Get global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    // Use registers for local accumulation
    float sum = 0.0f;
    
    // Unroll loop for better instruction-level parallelism
    #pragma unroll 4
    while (idx < n) {
        // Coalesced memory access pattern
        float log_pred = __ldg(&log_predictions[idx]); // Use read-only cache
        float target = __ldg(&targets[idx]);
        sum += expf(log_pred) - target * log_pred;
        idx += blockDim.x * gridDim.x;
    }
    
    // Single write to shared memory
    partial_sums[threadIdx.x] = sum;
    __syncthreads();
    
    // Warp-level reduction first (no sync needed within warp)
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Write warp result to shared memory
    if (threadIdx.x % warpSize == 0) {
        partial_sums[threadIdx.x / warpSize] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (threadIdx.x < warpSize) {
        sum = (threadIdx.x < blockDim.x/warpSize) ? partial_sums[threadIdx.x] : 0.0f;
        
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (threadIdx.x == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize block size to multiple of warp size
    const int threads = 256;  // 8 warps per block
    const int blocks = min(65535, (n + threads - 1) / threads); // Limit max blocks
    const int shared_mem = (threads/32) * sizeof(float); // Reduced shared memory
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}