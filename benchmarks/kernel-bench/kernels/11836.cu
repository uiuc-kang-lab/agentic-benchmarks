#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* log_predictions,
    const float* targets, 
    float* output,
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
        sum += expf(log_pred) - target * log_pred;
        idx += blockDim.x * gridDim.x;
    }
    
    // Store in shared memory
    partial_sums[threadIdx.x] = sum;
    __syncthreads();
    
    // Parallel reduction using shared memory for larger blocks
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction for final 32 elements
    if (threadIdx.x < 32) {
        volatile float* smem = partial_sums;
        if (blockDim.x >= 64) smem[threadIdx.x] += smem[threadIdx.x + 32];
        
        // Using warp-level primitives for final reduction
        float warp_sum = smem[threadIdx.x];
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 16);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 8);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 4);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 2);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 1);
        
        if (threadIdx.x == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
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