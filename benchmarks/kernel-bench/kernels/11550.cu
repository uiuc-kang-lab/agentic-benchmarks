#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void balanced_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int wid = tid >> 5;  // Warp ID
    const int lid = tid & 31;  // Lane ID
    const int warps_per_block = blockDim.x >> 5;
    
    // Calculate work per thread
    const int total_threads = gridDim.x * blockDim.x;
    const int elements_per_thread = (n + total_threads - 1) / total_threads;
    const int thread_start = (blockIdx.x * blockDim.x + tid) * elements_per_thread;
    const int thread_end = min(thread_start + elements_per_thread, n);
    
    // Shared memory for partial sums - only need one float per warp
    __shared__ float warp_sums[32];  // Support up to 32 warps
    
    // Local accumulator
    float sum = 0.0f;
    
    // Process assigned elements in chunks of 4 for better memory coalescing
    #pragma unroll 4
    for (int idx = thread_start; idx < thread_end; idx++) {
        sum += compute_kl_div(log_predictions[idx], targets[idx]);
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    
    // Store warp results
    if (lid == 0) {
        warp_sums[wid] = sum;
    }
    __syncthreads();
    
    // Final reduction using first warp
    if (wid == 0) {
        float warp_sum = 0.0f;
        if (lid < warps_per_block) {
            warp_sum = warp_sums[lid];
        }
        
        // Final warp reduction
        warp_sum = warp_reduce_sum(warp_sum);
        
        // Add block result to global sum
        if (lid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize launch configuration for better occupancy
    const int threads_per_block = 256;  // Use 256 threads for better occupancy
    // Calculate optimal number of blocks based on SM count
    const int num_blocks = min((n + threads_per_block - 1) / threads_per_block, 
                             32 * 108); // Assuming H100 has ~108 SMs
    
    balanced_kl_div_kernel<<<num_blocks, threads_per_block, 0>>>(
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