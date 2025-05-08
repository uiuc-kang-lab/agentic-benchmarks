#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for KL divergence calculation with warp-level optimizations
__global__ void kl_div_kernel_warp_optimized(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {
    
    // Get global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_size = 32;
    const unsigned int warp_id = threadIdx.x / warp_size;
    const unsigned int lane_id = threadIdx.x % warp_size;
    const unsigned int warps_per_block = blockDim.x / warp_size;
    
    // Shared memory for partial sums (one per warp)
    extern __shared__ float warp_sums[];
    
    float sum = 0.0f;
    
    // Calculate KL divergence for this thread's elements
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
        idx += blockDim.x * gridDim.x;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    
    __syncthreads();
    
    // First warp reduces results of all warps
    if (warp_id == 0 && lane_id < warps_per_block) {
        float warp_sum = warp_sums[lane_id];
        
        // Warp-level reduction of the final results
        #pragma unroll
        for (int offset = warps_per_block/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Only the first thread in the block does the atomic add
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward_warp(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int warps_per_block = threads / 32;
    const int shared_mem = warps_per_block * sizeof(float);
    
    kl_div_kernel_warp_optimized<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_warp, "KL divergence forward with warp optimization (CUDA)");
}