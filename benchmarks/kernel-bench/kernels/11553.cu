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

template<int BLOCK_SIZE>
__global__ void optimized_modular_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Ensure aligned access within warps
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int gid = blockIdx.x * BLOCK_SIZE + tid;
    
    // Shared memory for partial sums - one element per thread
    extern __shared__ float shared_mem[];
    
    // Local accumulator
    float thread_sum = 0.0f;
    
    // Process elements with vectorized grid stride loop
    #pragma unroll 4
    for (int idx = gid; idx < n; idx += gridDim.x * BLOCK_SIZE) {
        if (idx < n) {  // Boundary check
            thread_sum += compute_kl_div(log_predictions[idx], targets[idx]);
        }
    }
    
    // Warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);
    
    // Write warp results to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        float sum = (tid < BLOCK_SIZE/32) ? shared_mem[tid] : 0.0f;
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
    
    // Launch parameters
    constexpr int BLOCK_SIZE = 256;  // Must be multiple of 32
    const int blocks = min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 1024);
    const int shared_mem = (BLOCK_SIZE/32) * sizeof(float);
    
    optimized_modular_kl_div_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, shared_mem>>>(
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