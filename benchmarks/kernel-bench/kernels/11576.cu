#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute KL divergence for a single element
__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Device function for warp-level reduction without synchronization
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_reduced_sync(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int wid = tid / warpSize;
    const int lane = tid % warpSize;
    
    // Shared memory for partial sums - one element per warp plus one final result
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    // Compute local sum without synchronization
    #pragma unroll 4
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        sum += compute_kl_div(log_predictions[i], targets[i]);
    }
    
    // Warp-level reduction without synchronization
    sum = warp_reduce_sum(sum);
    
    // Only the first thread in each warp writes to shared memory
    if (lane == 0) {
        partial_sums[wid] = sum;
    }
    
    // Single synchronization point to ensure all warp results are visible
    __syncthreads();
    
    // Final reduction across warps - only first warp needs to work
    if (wid == 0) {
        // Load this thread's value from shared memory if it's within bounds
        float warp_sum = (tid < (blockDim.x + warpSize - 1) / warpSize) ? partial_sums[tid] : 0.0f;
        
        // Final warp reduction
        warp_sum = warp_reduce_sum(warp_sum);
        
        // First thread in block adds to global result
        if (lane == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward_reduced_sync(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    // Shared memory size = number of warps per block * sizeof(float)
    const int shared_mem = (threads + warpSize - 1) / warpSize * sizeof(float);
    
    kl_div_kernel_reduced_sync<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_reduced_sync, "KL divergence forward with reduced synchronization (CUDA)");
}