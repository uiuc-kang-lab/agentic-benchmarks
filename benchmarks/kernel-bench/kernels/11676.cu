#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel with different block sizes to determine optimal configuration
__global__ void block_size_experiment_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Constants for warp operations
    const unsigned int warp_size = 32;
    const unsigned int lane_id = threadIdx.x % warp_size;
    const unsigned int warp_id = threadIdx.x / warp_size;
    const unsigned int warps_per_block = blockDim.x / warp_size;
    
    // Shared memory only needed for final warp reduction
    extern __shared__ float warp_results[];
    
    float sum = 0.0f;
    
    // Calculate global index and stride
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process elements with stride pattern
    for (int i = tid; i < n; i += stride) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction across warps (only first warp)
    if (warp_id == 0) {
        // Load warp result or zero if lane is beyond number of warps
        float warp_sum = (lane_id < warps_per_block) ? warp_results[lane_id] : 0.0f;
        
        // Final warp-level reduction
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // First thread adds to global output
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor block_size_experiment_kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimal block sizes for different scenarios
    int optimal_threads_per_block = 256;  // Default value
    const int block_sizes[] = {32, 64, 128, 256, 512};
    
    for (int block_size : block_sizes) {
        if (n % block_size == 0) {
            optimal_threads_per_block = block_size;
            break;
        }
    }

    const int num_warps = optimal_threads_per_block / 32;
    const int blocks = min(256, (n + optimal_threads_per_block - 1) / optimal_threads_per_block);
    const int shared_mem = num_warps * sizeof(float);
    
    block_size_experiment_kl_div_kernel<<<blocks, optimal_threads_per_block, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_size_experiment_kl_div_cuda_forward, "Block size experiment KL divergence forward (CUDA)");
}