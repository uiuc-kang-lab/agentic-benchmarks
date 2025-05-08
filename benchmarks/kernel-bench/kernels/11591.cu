#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute KL divergence for a single element
__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Kernel for KL divergence calculation with optimized memory access
__global__ void kl_div_kernel_memory_access_optimized(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Get thread indices
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    // Optimize global memory access using __ldg for read-only data
    #pragma unroll 4
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        float log_pred = __ldg(&log_predictions[i]);
        float target = __ldg(&targets[i]);
        sum += compute_kl_div(log_pred, target);
    }
    
    // Store in shared memory
    partial_sums[tid] = sum;
    __syncthreads();
    
    // Perform block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    // Use atomic operation only once per block
    if (tid == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward_memory_access_optimized(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel_memory_access_optimized<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_memory_access_optimized, "KL divergence forward with memory access optimization (CUDA)");
}