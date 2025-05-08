#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void optimized_kl_div_kernel(
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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Vector loading for better memory coalescing
    float2 cache;
    for (int i = tid; i < n/2; i += stride) {
        const float2* log_pred_ptr = reinterpret_cast<const float2*>(log_predictions + 2*i);
        const float2* target_ptr = reinterpret_cast<const float2*>(targets + 2*i);
        
        cache = __ldg(log_pred_ptr);
        float2 target_val = __ldg(target_ptr);
        
        sum += expf(cache.x) - target_val.x * cache.x;
        sum += expf(cache.y) - target_val.y * cache.y;
    }
    
    // Handle remaining elements
    if (tid < n && tid >= n/2*2) {
        float log_pred = __ldg(&log_predictions[tid]);
        float target = __ldg(&targets[tid]);
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
    
    // Final reduction across warps
    if (warp_id == 0 && lane_id < warps_per_block) {
        float warp_sum = warp_results[lane_id];
        
        #pragma unroll
        for (int offset = (warps_per_block + 1)/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor optimized_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Dynamic block size selection based on input size
    int block_size = 256;  // Default
    if (n > 65536) block_size = 512;
    else if (n < 8192) block_size = 128;
    
    const int max_blocks = 256;
    const int num_blocks = min(max_blocks, (n + block_size - 1) / block_size);
    const int shared_mem = (block_size/32) * sizeof(float);
    
    optimized_kl_div_kernel<<<num_blocks, block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_kl_div_forward, "Optimized KL divergence forward (CUDA)");
}