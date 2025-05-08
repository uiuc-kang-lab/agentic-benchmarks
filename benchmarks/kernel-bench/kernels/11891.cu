#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_sync_optimized(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n,
    const int elements_per_thread) {
    
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int global_thread_id = blockIdx.x * blockDim.x + tid;
    
    extern __shared__ float warp_results[];
    
    float thread_sum = 0.0f;
    
    // Each thread processes multiple elements
    const int start_idx = global_thread_id * elements_per_thread;
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int idx = start_idx + i;
        if (idx < n) {
            const float log_pred = log_predictions[idx];
            const float target = targets[idx];
            thread_sum += expf(log_pred) - target * log_pred;
        }
    }
    
    // First level reduction within warps
    thread_sum = warp_reduce(thread_sum);
    
    // Store warp results only once per warp
    if (lane == 0) {
        warp_results[wid] = thread_sum;
    }
    
    // Synchronize only when necessary
    __syncthreads();
    
    // Final reduction across warps
    if (wid == 0) {
        float warp_sum = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_results[lane] : 0.0f;
        warp_sum = warp_reduce(warp_sum);
        
        if (lane == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Calculate optimal grid dimensions
    const int elements_per_thread = ELEMENTS_PER_THREAD;
    const int total_threads_needed = (n + elements_per_thread - 1) / elements_per_thread;
    const int blocks = (total_threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Shared memory for warp results
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int shared_mem = warps_per_block * sizeof(float);
    
    kl_div_kernel_sync_optimized<<<blocks, BLOCK_SIZE, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        elements_per_thread
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA sync optimized)");
}