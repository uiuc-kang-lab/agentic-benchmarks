#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_size = 32;
    const unsigned int warp_id = tid / warp_size;
    const unsigned int lane_id = tid % warp_size;
    const unsigned int grid_size = blockDim.x * gridDim.x;
    
    // Align processing to warp size
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Main processing loop - uniform across warp
    #pragma unroll 4
    while (idx < (n & ~(warp_size - 1))) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
        idx += grid_size;
    }
    
    // Handle remaining elements - still maintaining warp uniformity
    if (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction - no divergent branches
    sum = warp_reduce_sum(sum);
    
    // Shared memory for warp results
    __shared__ float warp_sums[32];  // Maximum number of warps per block
    
    // Store warp results - uniform write
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction by first warp only
    if (warp_id == 0) {
        // Load value or 0 - uniform across warp
        sum = (lane_id < (blockDim.x / warp_size)) ? warp_sums[lane_id] : 0.0f;
        
        // Final warp reduction - no divergent branches
        sum = warp_reduce_sum(sum);
        
        // Single thread atomic add
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Align block size to warp size for uniform execution
    const int threads_per_block = 256;  // 8 warps per block
    const int num_blocks = min(256, (n + threads_per_block - 1) / threads_per_block);
    
    kl_div_kernel<<<num_blocks, threads_per_block, sizeof(float) * (threads_per_block/32)>>>(
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