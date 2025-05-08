#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel_1d_grid(const float* log_predictions, const float* targets, float* output, const int n) {
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
        idx += blockDim.x * gridDim.x;
    }
    
    partial_sums[threadIdx.x] = sum;
    __syncthreads();

    // Initial block reduction
    for (int stride = blockDim.x/2; stride >= 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Warp-level reduction for final 32 elements
    if (threadIdx.x < 32) {
        float val = partial_sums[threadIdx.x];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        
        if (threadIdx.x == 0)
            atomicAdd(output, val);
    }
}

torch::Tensor kl_div_cuda_forward_1d_grid(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel_1d_grid<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_1d_grid, "KL divergence forward 1D grid (CUDA)");
}