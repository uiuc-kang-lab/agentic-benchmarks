#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int grid_stride = blockDim.x * gridDim.x;
    
    // Shared memory for final reduction only
    extern __shared__ float partial_sums[];
    
    // Local accumulation without synchronization
    float local_sum = 0.0f;
    
    // Each thread accumulates multiple elements without synchronization
    for (int idx = gid; idx < n; idx += grid_stride) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        local_sum += expf(log_pred) - target * log_pred;
    }
    
    // Store local sum in shared memory - single sync point
    partial_sums[tid] = local_sum;
    __syncthreads();
    
    // Single reduction phase with minimal syncs
    for (int stride = blockDim.x/2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction without sync
    if (tid < 32) {
        // Unrolled warp reduction - no sync needed within a warp
        if (blockDim.x >= 64) partial_sums[tid] += partial_sums[tid + 32];
        if (blockDim.x >= 32) partial_sums[tid] += partial_sums[tid + 16];
        if (blockDim.x >= 16) partial_sums[tid] += partial_sums[tid + 8];
        if (blockDim.x >= 8) partial_sums[tid] += partial_sums[tid + 4];
        if (blockDim.x >= 4) partial_sums[tid] += partial_sums[tid + 2];
        if (blockDim.x >= 2) partial_sums[tid] += partial_sums[tid + 1];
        
        // Only first thread in block writes result
        if (tid == 0) {
            atomicAdd(output, partial_sums[0]);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);
    
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