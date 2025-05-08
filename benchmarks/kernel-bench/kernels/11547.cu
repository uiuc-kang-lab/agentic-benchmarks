#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256

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

__global__ void thread_optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    extern __shared__ float shared_mem[];
    float* partial_sums = shared_mem;
    
    // Calculate global thread index
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float thread_sum = 0.0f;
    
    // Use grid-stride loop for coalesced global memory access
    for (int idx = global_idx; idx < n; idx += stride) {
        thread_sum += compute_kl_div(log_predictions[idx], targets[idx]);
    }
    
    // Warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);
    
    // Store warp results
    if (threadIdx.x % 32 == 0) {
        partial_sums[threadIdx.x / 32] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (threadIdx.x < (blockDim.x / 32)) {
        float sum = partial_sums[threadIdx.x];
        sum = warp_reduce_sum(sum);
        
        if (threadIdx.x == 0) {
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
    const int threads = 256; // Multiple of warp size
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = (threads / 32) * sizeof(float);
    
    thread_optimized_kl_div_kernel<<<blocks, threads, shared_mem>>>(
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