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

__global__ void kl_div_kernel_tuned_blocksize(
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
    #pragma unroll 8
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        sum += compute_kl_div(log_predictions[i], targets[i]);
    }
    
    // Store in shared memory
    partial_sums[tid] = sum;
    __syncthreads();
    
    // Perform block-level reduction
    if (blockDim.x >= 128) { if (tid < 64) { partial_sums[tid] += partial_sums[tid + 64]; } __syncthreads(); }
    if (tid < 32) {
        if (blockDim.x >= 64) partial_sums[tid] += partial_sums[tid + 32];
        // Final warp reduction using shuffle
        float warp_sum = partial_sums[tid];
        warp_sum = warp_reduce_sum(warp_sum);
        if (tid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward_tuned_blocksize(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimized block size of 128 threads
    const int threads = 128;
    
    // Calculate optimal number of blocks based on device properties
    int max_blocks_per_sm;
    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        kl_div_kernel_tuned_blocksize,
        threads,
        threads * sizeof(float));
    
    const int optimal_blocks = min(
        (n + threads - 1) / threads,
        max_blocks_per_sm * num_sm
    );
    
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel_tuned_blocksize<<<optimal_blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_tuned_blocksize, "KL divergence forward with tuned block size (CUDA)");
}