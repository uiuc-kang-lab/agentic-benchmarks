#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Aligned memory access for better coalescing
#define ALIGN_SIZE 32

__global__ void optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Calculate aligned index for this thread
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_threads = blockDim.x;
    const int warp_size = 32;
    
    // Ensure aligned access within warps
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;
    const int aligned_idx = bid * num_threads + warp_id * warp_size + lane_id;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    partial_sums[tid] = 0.0f;
    
    // Process elements with stride of complete warps
    for (int i = aligned_idx; i < n; i += gridDim.x * num_threads) {
        if (i < n) {
            float log_pred = log_predictions[i];
            float target = targets[i];
            partial_sums[tid] += expf(log_pred) - target * log_pred;
        }
    }
    __syncthreads();
    
    // Warp-level reduction first
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, partial_sums[tid], offset);
        if (lane_id < offset) {
            partial_sums[tid] += other;
        }
    }
    
    // Block-level reduction using shared memory
    if (lane_id == 0) {
        float warp_sum = partial_sums[tid];
        int warp_idx = warp_id;
        __syncthreads();
        
        partial_sums[warp_idx] = warp_sum;
        __syncthreads();
        
        // Final reduction by first thread
        if (tid == 0) {
            float block_sum = 0.0f;
            for (int i = 0; i < num_threads/warp_size; i++) {
                block_sum += partial_sums[i];
            }
            atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters optimized for memory coalescing
    const int threads = 256; // Multiple of warp size (32)
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    // Launch kernel
    optimized_kl_div_kernel<<<blocks, threads, shared_mem>>>(
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