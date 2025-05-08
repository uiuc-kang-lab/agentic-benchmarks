#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for KL divergence calculation with warp-level primitives
__global__ void kl_div_kernel_warp_optimized(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {
    
    // Get global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Calculate KL divergence for this thread's elements
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        // F.kl_div implementation:
        // output = exp(log_predictions) - targets * log_predictions
        float log_pred = log_predictions[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction using shuffle operations
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Write result for this warp to global memory
    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, sum);
    }
}

torch::Tensor kl_div_cuda_forward_warp_optimized(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    // Get tensor sizes
    const int n = log_predictions.numel();
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    // Launch kernel
    kl_div_kernel_warp_optimized<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_warp_optimized, "KL divergence forward with warp optimization (CUDA)");
}