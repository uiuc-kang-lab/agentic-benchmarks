#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute KL divergence for a single element
__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Device function for warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Device function for block-level reduction
__device__ __forceinline__ void block_reduce_sum(float* shared_data, int tid) {
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Final warp reduction
    if (tid < 32) {
        float warp_sum = shared_data[tid];
        if (tid + 32 < blockDim.x) {
            warp_sum += shared_data[tid + 32];
        }
        warp_sum = warp_reduce_sum(warp_sum);
        shared_data[tid] = warp_sum;
    }
}

__global__ void kl_div_kernel_modular_optimized(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Get thread indices
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    // Calculate KL divergence for this thread's elements
    #pragma unroll 4
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        sum += compute_kl_div(log_predictions[i], targets[i]);
    }
    
    // Store in shared memory
    partial_sums[tid] = sum;
    __syncthreads();
    
    // Perform block-level reduction
    block_reduce_sum(partial_sums, tid);
    
    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward_modular_optimized(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel_modular_optimized<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_modular_optimized, "KL divergence forward with modular optimizations (CUDA)");
}