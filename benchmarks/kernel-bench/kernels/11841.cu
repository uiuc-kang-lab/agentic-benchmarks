#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kldiv_uniform_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int64_t n) {
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_id = threadIdx.x / 32;
    
    __shared__ float warp_results[32];
    
    // Each thread processes multiple elements with uniform stride
    float thread_sum = 0.0f;
    
    #pragma unroll 4
    for (int64_t i = tid; i < n; i += stride) {
        const float log_pred = log_predictions[i];
        const float target = targets[i];
        thread_sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction - all threads participate
    float warp_sum = warp_reduce_sum(thread_sum);
    
    // First thread in each warp writes result
    if (lane_id == 0) {
        warp_results[warp_id] = warp_sum;
    }
    
    __syncthreads();
    
    // Final reduction across warps - done by first warp
    if (warp_id == 0) {
        // Load warp result or zero based on validity
        const unsigned int num_warps = (blockDim.x + 31) / 32;
        float final_sum = (lane_id < num_warps) ? warp_results[lane_id] : 0.0f;
        
        // Reduce within first warp
        final_sum = warp_reduce_sum(final_sum);
        
        // Single thread adds to global result
        if (lane_id == 0) {
            atomicAdd(output, final_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {
    
    const int64_t n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    kldiv_uniform_kernel<<<blocks, threads>>>(
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