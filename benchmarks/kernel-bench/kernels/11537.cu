#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device functions for modular kernel components
__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void optimized_modular_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Calculate aligned index for coalesced memory access
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_threads = blockDim.x;
    const int warp_size = 32;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;
    const int aligned_idx = bid * num_threads + warp_id * warp_size + lane_id;
    
    // Shared memory for partial sums - only need warp-level storage
    extern __shared__ float partial_sums[];
    
    // Local accumulator for better register utilization
    float thread_sum = 0.0f;
    
    // Process elements with aligned access and grid stride
    for (int i = aligned_idx; i < n; i += gridDim.x * num_threads) {
        if (i < n) {
            thread_sum += compute_kl_div(log_predictions[i], targets[i]);
        }
    }
    
    // Warp-level reduction using shuffle operations
    thread_sum = warp_reduce_sum(thread_sum);
    
    // Store warp results in shared memory
    if (lane_id == 0) {
        partial_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        float sum = (tid < blockDim.x/warp_size) ? partial_sums[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        
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
    
    // Optimized launch parameters
    const int threads = 256; // Multiple of warp size
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = (threads/32) * sizeof(float);
    
    optimized_modular_kl_div_kernel<<<blocks, threads, shared_mem>>>(
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