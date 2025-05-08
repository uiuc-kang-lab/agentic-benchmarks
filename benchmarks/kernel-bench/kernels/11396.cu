#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* log_predictions,
    const float* targets, 
    float* block_results,
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
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        block_results[blockIdx.x] = partial_sums[0];
    }
}

__global__ void final_reduction_kernel(float* block_results, float* output, int num_blocks) {
    extern __shared__ float partial_sums[];
    
    int idx = threadIdx.x;
    float sum = 0.0f;
    
    while (idx < num_blocks) {
        sum += block_results[idx];
        idx += blockDim.x;
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
        output[0] = partial_sums[0];
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    auto options = torch::TensorOptions().device(torch::kCUDA);
    auto block_results = torch::empty({blocks}, options);
    auto output = torch::zeros({1}, options);
    
    kl_div_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        n
    );
    
    final_reduction_kernel<<<1, threads, threads * sizeof(float)>>>(
        block_results.data_ptr<float>(),
        output.data_ptr<float>(),
        blocks
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}