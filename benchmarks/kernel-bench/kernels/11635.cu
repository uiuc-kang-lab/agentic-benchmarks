#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ float const_log_predictions[1024]; // Assuming maximum size that fits within constant memory

// CUDA kernel for KL divergence calculation 
__global__ void kl_div_kernel_constant_memory(
    const float* targets, 
    float* output,
    const int n) {
    
    // Get global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    // Calculate KL divergence for this thread's elements
    while (idx < n) {
        // F.kl_div implementation:
        // output = exp(log_predictions) - targets * log_predictions
        float log_pred = const_log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
        
        idx += blockDim.x * gridDim.x;
    }
    
    // Store in shared memory
    partial_sums[threadIdx.x] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
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

torch::Tensor kl_div_cuda_forward_constant_memory(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    // Get tensor sizes
    const int n = log_predictions.numel();
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Copy log_predictions to constant memory
    cudaMemcpyToSymbol(const_log_predictions, log_predictions.data_ptr<float>(), n * sizeof(float));
    
    // Launch parameters
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    // Launch kernel
    kl_div_kernel_constant_memory<<<blocks, threads, shared_mem>>>(
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_constant_memory, "KL divergence forward with constant memory (CUDA)");
}