#include <torch/extension.h>
#include <cuda_runtime.h>

__constant__ float c_log_predictions[16384]; // 64KB/(4B*2) for two buffers
__constant__ float c_targets[16384];

__global__ void kl_div_constant_kernel(
    float* output,
    const int n) {
    
    extern __shared__ float partial_sums[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    while (idx < n) {
        float log_pred = c_log_predictions[idx];
        float target = c_targets[idx];
        sum += expf(log_pred) - target * log_pred;
        idx += blockDim.x * gridDim.x;
    }

    partial_sums[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
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
    TORCH_CHECK(n <= 16384, "Input size exceeds constant memory capacity");
    
    // Copy inputs to constant memory
    cudaMemcpyToSymbol(c_log_predictions, log_predictions.data_ptr<float>(), n * sizeof(float));
    cudaMemcpyToSymbol(c_targets, targets.data_ptr<float>(), n * sizeof(float));

    auto output = torch::zeros({1}, log_predictions.options());
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    kl_div_constant_kernel<<<blocks, threads, shared_mem>>>(
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL Divergence with constant memory (CUDA)");
}
