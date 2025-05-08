#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for computing KL divergence element
__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Device function for parallel reduction
__device__ __forceinline__ void reduce_block(volatile float* partial_sums, unsigned int tid) {
    if (blockDim.x >= 64) partial_sums[tid] += partial_sums[tid + 32];
    if (blockDim.x >= 32) partial_sums[tid] += partial_sums[tid + 16];
    if (blockDim.x >= 16) partial_sums[tid] += partial_sums[tid + 8];
    if (blockDim.x >= 8) partial_sums[tid] += partial_sums[tid + 4];
    if (blockDim.x >= 4) partial_sums[tid] += partial_sums[tid + 2];
    if (blockDim.x >= 2) partial_sums[tid] += partial_sums[tid + 1];
}

// Device function for block-wide reduction
__device__ void block_reduce(float* partial_sums, unsigned int tid) {
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    // Final warp reduction
    if (tid < 32) reduce_block(partial_sums, tid);
}

// Main CUDA kernel
__global__ void kl_div_kernel_modular(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Get thread ID
    const unsigned int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    // Compute local sum using stride loop
    float sum = 0.0f;
    for (; idx < n; idx += stride) {
        sum += compute_kl_element(log_predictions[idx], targets[idx]);
    }
    
    // Store in shared memory
    partial_sums[tid] = sum;
    __syncthreads();
    
    // Perform reduction
    block_reduce(partial_sums, tid);
    
    // Write result
    if (tid == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel_modular<<<blocks, threads, shared_mem>>>(
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
