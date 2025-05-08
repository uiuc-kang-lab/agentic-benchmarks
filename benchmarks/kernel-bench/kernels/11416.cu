#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31;
    const unsigned int grid_stride = gridDim.x * blockDim.x;
    
    // Shared memory for warp results
    __shared__ float warp_results[8]; // Assuming block size 256 = 8 warps
    
    // Initialize accumulator
    float sum = 0.0f;
    
    // Main processing loop - all threads in warp execute same number of iterations
    const unsigned int start_idx = bid * blockDim.x + tid;
    #pragma unroll 4
    for (unsigned int idx = start_idx; idx < n; idx += grid_stride) {
        const float log_pred = log_predictions[idx];
        const float target = targets[idx];
        const bool valid = idx < n;
        // Use predication instead of branching
        sum += valid ? (expf(log_pred) - target * log_pred) : 0.0f;
    }
    
    // Warp-level reduction - no divergence as all threads participate
    sum = warp_reduce(sum);
    
    // Store warp results - only one thread per warp writes
    const bool is_warp_leader = (lane_id == 0);
    if (is_warp_leader) {
        warp_results[warp_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction using first warp
    if (warp_id == 0) {
        // Load warp result or 0 based on lane ID
        float warp_sum = (lane_id < (blockDim.x >> 5)) ? warp_results[lane_id] : 0.0f;
        
        // Final warp reduction
        warp_sum = warp_reduce(warp_sum);
        
        // Single atomic add per block
        const bool is_first_thread = (tid == 0);
        if (is_first_thread) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch configuration
    const int threads = 256;
    const int blocks = min(256, (n + threads - 1) / threads);
    const int shared_mem = (threads / 32) * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
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