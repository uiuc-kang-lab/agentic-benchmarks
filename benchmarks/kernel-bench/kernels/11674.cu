#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for computing KL divergence element
__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Warp reduction using shuffle instructions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kl_div_kernel_strided(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Calculate thread and block indices
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int warp_id = tid / warpSize;
    const unsigned int lane = tid % warpSize;
    
    // Calculate grid stride for efficient processing of large arrays
    const unsigned int grid_stride = blockDim.x * gridDim.x;
    unsigned int idx = bid * blockDim.x + tid;
    
    // Shared memory for partial sums (one per warp)
    extern __shared__ float warp_sums[];
    
    float thread_sum = 0.0f;
    
    // Grid stride loop - each thread processes multiple elements
    while (idx < n) {
        // Process current element
        thread_sum += compute_kl_element(log_predictions[idx], targets[idx]);
        
        // Move to next element using grid stride
        idx += grid_stride;
    }
    
    // Perform warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);
    
    // First thread in each warp writes the result to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    
    __syncthreads();
    
    // Final reduction across warps (performed by first warp)
    if (warp_id == 0) {
        float warp_sum = 0.0f;
        
        // Number of warps in the block
        const unsigned int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        
        // Each thread in first warp handles one warp's sum
        if (lane < num_warps) {
            warp_sum = warp_sums[lane];
        }
        
        // Final warp reduction
        warp_sum = warp_reduce_sum(warp_sum);
        
        // First thread in block writes final sum
        if (lane == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize launch configuration
    const int threads_per_block = 256;  // Multiple of warp size (32)
    const int max_blocks = 1024;
    const int num_blocks = min((n + threads_per_block - 1) / threads_per_block, max_blocks);
    
    // Calculate shared memory size (one float per warp)
    const int warps_per_block = (threads_per_block + warpSize - 1) / warpSize;
    const int shared_mem_size = warps_per_block * sizeof(float);
    
    kl_div_kernel_strided<<<num_blocks, threads_per_block, shared_mem_size>>>(
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