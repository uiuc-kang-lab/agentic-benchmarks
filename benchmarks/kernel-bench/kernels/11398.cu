#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for KL divergence calculation 
__global__ void kl_div_kernel(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {
    
    // Get global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    // Calculate KL divergence for this thread's elements
    // Unroll by 4 for better instruction-level parallelism
    #pragma unroll 4
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
        
        if (idx + blockDim.x < n) {
            log_pred = log_predictions[idx + blockDim.x];
            target = targets[idx + blockDim.x];
            sum += expf(log_pred) - target * log_pred;
        }
        
        if (idx + 2 * blockDim.x < n) {
            log_pred = log_predictions[idx + 2 * blockDim.x];
            target = targets[idx + 2 * blockDim.x];
            sum += expf(log_pred) - target * log_pred;
        }
        
        if (idx + 3 * blockDim.x < n) {
            log_pred = log_predictions[idx + 3 * blockDim.x];
            target = targets[idx + 3 * blockDim.x];
            sum += expf(log_pred) - target * log_pred;
        }
        
        idx += 4 * blockDim.x * gridDim.x;
    }
    
    // Store in shared memory
    partial_sums[threadIdx.x] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory with unrolled loop
    #pragma unroll
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    // Get tensor sizes
    const int n = log_predictions.numel();
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters
    const int threads = 256;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);
    const int shared_mem = threads * sizeof(float);
    
    // Launch kernel
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