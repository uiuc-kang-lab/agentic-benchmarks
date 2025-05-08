#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void optimized_kl_div_kernel(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {
    
    // Get global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    
    // Calculate KL divergence for this thread's elements
    for(int i = idx; i < n; i += stride) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Store in shared memory
    partial_sums[threadIdx.x] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (assumes warp size is 32)
    if (threadIdx.x < 32) {
        for (int s = 32; s > 0; s >>= 1) {
            partial_sums[threadIdx.x] += __shfl_down_sync(0xffffffff, partial_sums[threadIdx.x], s);
        }
    }
    
    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize thread and block count based on data size
    const int threads = 256;
    const int blocks = min(256, (n + threads - 1) / threads);
    const int shared_mem = threads * sizeof(float);
    
    optimized_kl_div_kernel<<<blocks, threads, shared_mem>>>(
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