#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 8

__device__ __forceinline__ float warp_reduce_shfl(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_warp(float val) {
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int wid = threadIdx.x / WARP_SIZE;
    
    // First reduce within warp
    val = warp_reduce_shfl(val);
    
    // Communicate results between warps using shuffle instructions
    if (lane == 0) {
        // Broadcast warp result to all threads in first warp
        val = __shfl_sync(0xffffffff, val, wid, WARP_SIZE);
    }
    
    // Final reduction in first warp
    if (wid == 0) {
        val = warp_reduce_shfl(val);
    }
    
    return val;
}

__global__ void kl_div_kernel_warp_optimized(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int global_tid = blockIdx.x * blockDim.x + tid;
    const int grid_stride = gridDim.x * blockDim.x;
    
    float thread_sum = 0.0f;
    
    // Process multiple elements per thread with grid stride loop
    #pragma unroll 2
    for (int idx = global_tid; idx < n; idx += grid_stride) {
        const float log_pred = log_predictions[idx];
        const float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
    }
    
    // Reduce within block using warp primitives
    thread_sum = block_reduce_warp(thread_sum);
    
    // Only one thread per block needs to do the atomic add
    if (tid == 0) {
        atomicAdd(output, thread_sum);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize grid dimensions for H100
    const int threads = BLOCK_SIZE;
    const int max_blocks = 512; // Adjusted for H100's capabilities
    const int blocks = min(max_blocks, (n + threads - 1) / threads);
    
    kl_div_kernel_warp_optimized<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA warp optimized)");
}