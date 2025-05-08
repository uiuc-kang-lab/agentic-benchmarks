#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void adaptive_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n,
    const int elements_per_thread) {
    
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    extern __shared__ float shared_mem[];
    float* warp_results = shared_mem;
    
    float thread_sum = 0.0f;
    
    // Coalesced memory access with multiple elements per thread
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int idx = gid * elements_per_thread + i;
        if (idx < n) {
            const float log_pred = log_predictions[idx];
            const float target = targets[idx];
            thread_sum += expf(log_pred) - target * log_pred;
        }
    }
    
    // Grid-stride loop for remaining elements
    for (int idx = gid * elements_per_thread + elements_per_thread; 
         idx < n; 
         idx += gridDim.x * blockDim.x * elements_per_thread) {
        const float log_pred = log_predictions[idx];
        const float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
    }
    
    // Two-level reduction: first within warps
    thread_sum = warp_reduce(thread_sum);
    
    if (lane == 0) {
        warp_results[wid] = thread_sum;
    }
    __syncthreads();
    
    // Then across warps
    if (wid == 0) {
        float warp_sum = (lane < (blockDim.x / WARP_SIZE)) ? warp_results[lane] : 0.0f;
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
    
    // Adaptive block size and elements per thread based on input size
    const int elements_per_thread = ELEMENTS_PER_THREAD;
    const int block_size = MAX_BLOCK_SIZE;
    const int total_threads_needed = (n + elements_per_thread - 1) / elements_per_thread;
    const int num_blocks = min(
        (total_threads_needed + block_size - 1) / block_size,
        65535  // Maximum number of blocks
    );
    
    const int warps_per_block = block_size / WARP_SIZE;
    const int shared_mem_size = warps_per_block * sizeof(float);
    
    adaptive_kl_div_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        elements_per_thread
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA adaptive)");
}