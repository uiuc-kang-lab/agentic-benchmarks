#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel for KL divergence calculation
template<int BLOCK_SIZE>
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Thread and block identification with memory alignment
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * BLOCK_SIZE + tid;
    const int grid_stride = gridDim.x * BLOCK_SIZE;
    
    // Warp-level variables
    const int warp_size = 32;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;
    
    // Shared memory for partial sums
    __shared__ float partial_sums[BLOCK_SIZE];
    float thread_sum = 0.0f;
    
    // Process elements with vectorized loads where possible
    #pragma unroll 4
    for (int i = gid; i < n; i += grid_stride) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        thread_sum += __expf(log_pred) - target * log_pred;
    }
    
    // Store initial sum
    partial_sums[tid] = thread_sum;
    __syncthreads();
    
    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // Write warp results to shared memory
    if (lane_id == 0) {
        partial_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (tid < warp_size) {
        thread_sum = (tid < (BLOCK_SIZE / warp_size)) ? partial_sums[tid] : 0.0f;
        
        // Warp-level reduction of final results
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        
        // Single thread writes final result
        if (tid == 0) {
            atomicAdd(output, thread_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters
    const int threads = 256;  // Must be power of 2 and multiple of warp size
    const int blocks = min(65535, (n + threads - 1) / threads);
    
    // Launch kernel with template parameter
    kl_div_kernel<256><<<blocks, threads>>>(
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