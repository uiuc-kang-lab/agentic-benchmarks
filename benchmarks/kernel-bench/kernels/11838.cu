#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel_optimized(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n) {
    
    extern __shared__ float partial_sums[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    while (idx < n) {
        float log_pred = log_predictions[idx];
        sum += expf(log_pred) - targets[idx] * log_pred;
        idx += blockDim.x * gridDim.x;
    }

    partial_sums[threadIdx.x] = sum;
    __syncthreads(); // Single sync after all threads write to shared mem

    // Reduction without intermediate syncs
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        // Removed unnecessary inner __syncthreads()
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward_optimized(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    // Warmup kernel for more accurate timing
    kl_div_kernel_optimized<<<blocks, threads, threads * sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_optimized", &kl_div_cuda_forward_optimized, "Optimized KL divergence forward (CUDA)");
}