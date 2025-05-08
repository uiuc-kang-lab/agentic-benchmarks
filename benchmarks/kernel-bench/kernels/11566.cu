#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel_warp_shfl(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Get thread indices
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid % 32;
    const unsigned int warp_id = tid / 32;
    const unsigned int global_idx = blockIdx.x * blockDim.x + tid;
    
    float sum = 0.0f;
    
    // Process elements with unrolled loop
    #pragma unroll 4
    for (int i = global_idx; i < n; i += blockDim.x * gridDim.x) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    __shared__ float warp_sums[32]; // Maximum 32 warps per block
    
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp)
    if (warp_id == 0 && lane_id < (blockDim.x / 32)) {
        float warp_sum = warp_sums[lane_id];
        
        // Warp-level reduction for final results
        #pragma unroll
        for (int offset = (blockDim.x / 64); offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize launch parameters
    const int threads = 256; // 8 warps per block
    const int max_blocks = 128;
    const int blocks = min((n + threads - 1) / threads, max_blocks);
    
    kl_div_kernel_warp_shfl<<<blocks, threads>>>(
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