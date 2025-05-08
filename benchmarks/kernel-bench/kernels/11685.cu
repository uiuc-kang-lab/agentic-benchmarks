#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void minimal_sync_kl_div_kernel(
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
    
    for (int i = tid; i < n; i += stride) {
        float log_pred = __ldg(&log_predictions[i]);
        float target = __ldg(&targets[i]);
        sum += expf(log_pred) - target * log_pred;
    }
    
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        warp_results[warp_id] = sum;
    }
    
    // Synchronize only once before accessing shared memory
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

torch::Tensor minimal_sync_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Dynamic block size selection
    int best_block_size = 256;  // Default for H100
    const int max_potential_blocks = 256;
    
    if (n > 65536) best_block_size = 512;
    else if (n < 8192) best_block_size = 128;
    
    const int num_warps = best_block_size / 32;
    const int blocks = min(max_potential_blocks, (n + best_block_size - 1) / best_block_size);
    const int shared_mem = num_warps * sizeof(float);
    
    minimal_sync_kl_div_kernel<<<blocks, best_block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &minimal_sync_kl_div_forward, "KLDivLoss with minimal synchronization (CUDA)");
}
