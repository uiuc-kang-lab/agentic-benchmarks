#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* log_predictions,
    const float* targets, 
    float* block_results,
    float* final_output,
    const int n) {
    
    // Get global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    // Calculate KL divergence for this thread's elements
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        float exp_log_pred = expf(log_pred);
sum += exp_log_pred - target * log_pred;
        
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
    
    // Write block result to global array instead of atomic add
    if (threadIdx.x == 0) {
        block_results[blockIdx.x] = partial_sums[0];
    }
    
    // Only one thread from the first block performs final reduction
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float final_sum = 0.0f;
        for (int i = 0; i < gridDim.x; i++) {
            final_sum += block_results[i];
        }
        *final_output = final_sum;
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    // Allocate space for block results
    auto block_results = torch::zeros(blocks, log_predictions.options());
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}