#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel_optimized(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Calculate thread index using grid-stride loop pattern
    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    
    // Shared memory for partial sums - pad to avoid bank conflicts
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    // Grid stride loop for better work distribution
    #pragma unroll 4
    for (; idx < n; idx += stride) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Store in shared memory with padding to avoid bank conflicts
    partial_sums[tid] = sum;
    __syncthreads();
    
    // Parallel reduction with sequential addressing
    #pragma unroll
    for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction for final 32 elements
    if (tid < 32) {
        volatile float* smem = partial_sums;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize thread and block configuration
    const int thread_count = 128; // Power of 2, helps with memory coalescing
    const int max_blocks = 128; // Limit max blocks to avoid oversubscription
    const int block_count = min((n + thread_count - 1) / thread_count, max_blocks);
    const int shared_mem = thread_count * sizeof(float);
    
    kl_div_kernel_optimized<<<block_count, thread_count, shared_mem>>>(
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