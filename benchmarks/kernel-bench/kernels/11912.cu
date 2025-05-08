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

__global__ void kl_div_kernel_minimal_sync(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int global_thread_id = blockIdx.x * blockDim.x + tid;
    const int grid_stride = gridDim.x * blockDim.x;
    
    // Use registers for thread-local accumulation
    float thread_sum = 0.0f;
    
    // Process elements with grid stride loop
    for (int idx = global_thread_id; idx < n; idx += grid_stride) {
        const float log_pred = log_predictions[idx];
        const float target = targets[idx];
        thread_sum += __expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction (no synchronization needed within a warp)
    thread_sum = warp_reduce(thread_sum);
    
    // Only the first thread in each warp participates in the final reduction
    if (lane == 0) {
        // Use atomicAdd directly for warp results
        // This eliminates the need for shared memory and its associated synchronization
        atomicAdd(output, thread_sum);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Calculate optimal grid dimensions
    const int threads = BLOCK_SIZE;
    const int blocks = std::min(256, (n + threads * ELEMENTS_PER_THREAD - 1) / (threads * ELEMENTS_PER_THREAD));
    
    kl_div_kernel_minimal_sync<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA minimal sync)");
}