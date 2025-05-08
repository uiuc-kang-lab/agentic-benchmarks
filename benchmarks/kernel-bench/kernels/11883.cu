#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel_shared(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float shared_log_predictions[];
    extern __shared__ float shared_targets[];
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    // Load data into shared memory
    if (idx < n) {
        shared_log_predictions[threadIdx.x] = log_predictions[idx];
        shared_targets[threadIdx.x] = targets[idx];
    }
    __syncthreads();

    // Compute KL divergence using shared memory
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float log_pred = shared_log_predictions[i];
        float target = shared_targets[i];
        sum += expf(log_pred) - target * log_pred;
    }
    
    partial_sums[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = 3 * threads * sizeof(float); // Shared memory for log_predictions, targets, and partial_sums
    
    kl_div_kernel_shared<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA with shared memory optimization)");
}