#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {
    
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    extern __shared__ float partial_sums[];
    
    float sum[4] = {0};
    
    // Vectorized memory access for better memory throughput
    for (int i = 0; i < 4; ++i,
        idx += blockDim.x) {
        if (idx < n) {
            float log_pred = log_predictions[idx];
            float target = targets[idx];
            sum[i] += expf(log_pred) - target * log_pred;
        }
    }

    // Register accumulation before shared memory store
    float total = sum[0] + sum[1] + sum[2] + sum[3];
    partial_sums[threadIdx.x] = total;
    __syncthreads();

    // Uniform block reduction
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(output, partial_sums[0]);
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads*4 - 1) / (threads*4);
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
