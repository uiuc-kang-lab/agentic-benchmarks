#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel with improved memory access pattern
__global__ void optimized_kl_div_kernel_memory(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    
    extern __shared__ float warp_results[];
    
    float sum = 0.0f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Optimized memory access using __ldg and aligned vector loads
    for (int i = tid; i < n; i += stride) {
        float log_pred = __ldg(&log_predictions[i]);
        float target = __ldg(&targets[i]);
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        warp_results[warp_id] = sum;
    }
    
    __syncthreads();
    
    if (warp_id == 0) {
        float warp_sum = (lane_id < warps_per_block) ? warp_results[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor optimized_kl_div_cuda_forward_memory(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads_per_block = 256; // Keep as power of two for efficiency
    const int num_warps = threads_per_block / 32;
    const int blocks = min(256, (n + threads_per_block - 1) / threads_per_block);
    const int shared_mem = num_warps * sizeof(float);
    
    optimized_kl_div_kernel_memory<<<blocks, threads_per_block, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_kl_div_cuda_forward_memory, "Optimized KL divergence forward with memory enhancements (CUDA)");
}