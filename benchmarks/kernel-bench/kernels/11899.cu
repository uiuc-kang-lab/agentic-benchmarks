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

__global__ void kl_div_kernel_coalesced_memory(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int base_idx = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + tid;
    const int stride = BLOCK_SIZE;
    
    extern __shared__ float warp_results[];
    
    float thread_sum = 0.0f;
    
    // Ensure memory coalescing through aligned accesses
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = base_idx + i * stride;
        if (idx < n) {
            const float log_pred = log_predictions[idx];
            const float target = targets[idx];
            thread_sum += expf(log_pred) - target * log_pred;
        }
    }
    
    // First level reduction within warps
    thread_sum = warp_reduce(thread_sum);
    
    // Store warp results only once per warp
    if (tid % WARP_SIZE == 0) {
        warp_results[tid / WARP_SIZE] = thread_sum;
    }
    
    // Synchronize only when necessary
    __syncthreads();
    
    // Final reduction across warps
    if (tid < (BLOCK_SIZE / WARP_SIZE)) {
        float warp_sum = warp_results[tid];
        warp_sum = warp_reduce(warp_sum);
        
        if (tid == 0) {
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
    const int blocks = (n + BLOCK_SIZE * ELEMENTS_PER_THREAD - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD);
    
    // Shared memory for warp results
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int shared_mem = warps_per_block * sizeof(float);
    
    kl_div_kernel_coalesced_memory<<<blocks, BLOCK_SIZE, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA coalesced memory access)");}