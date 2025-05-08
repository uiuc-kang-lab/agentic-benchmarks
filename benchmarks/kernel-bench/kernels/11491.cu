#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel_even_workload(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {
    
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    // Vectorized memory access for better memory throughput
    for (int i = 0; i < 4; ++i, idx += blockDim.x) {
        if (idx < n) {
            float log_pred = log_predictions[idx];
            float target = targets[idx];
            sum += expf(log_pred) - target * log_pred;
        }
    }

    // Store sum in shared memory
    partial_sums[threadIdx.x] = sum;
    __syncthreads();

    // Uniform block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(output, partial_sums[0]);
}

torch::Tensor kl_div_cuda_forward_even_workload(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads*4 - 1) / (threads*4);
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel_even_workload<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_even_workload, "KL divergence forward even workload (CUDA)");
}