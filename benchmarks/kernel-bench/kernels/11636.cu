#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for KL divergence calculation using shared memory
__global__ void kl_div_kernel_shared_memory(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {
    
    // Get global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for partial sums and log_predictions
    extern __shared__ float shared_data[];
    float* partial_sums = shared_data;
    float* shared_log_predictions = &shared_data[blockDim.x];

    float sum = 0.0f;
    
    // Load log_predictions into shared memory
    if (idx < n) {
        shared_log_predictions[threadIdx.x] = log_predictions[idx];
    }
    __syncthreads();

    // Calculate KL divergence for this thread's elements
    while (idx < n) {
        // F.kl_div implementation:
        // output = exp(log_predictions) - targets * log_predictions
        float log_pred = shared_log_predictions[threadIdx.x];
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

torch::Tensor kl_div_cuda_forward_shared_memory(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    // Get tensor sizes
    const int n = log_predictions.numel();
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = 2 * threads * sizeof(float);
    
    // Launch kernel
    kl_div_kernel_shared_memory<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_shared_memory, "KL divergence forward with shared memory (CUDA)");
}