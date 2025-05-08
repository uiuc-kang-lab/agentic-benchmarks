#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int wid = tid >> 5;
    const unsigned int warps_per_block = blockDim.x >> 5;
    
    // Each thread processes 8 elements initially
    float thread_sum = 0.0f;
    int base_idx = (blockIdx.x * blockDim.x + tid) * 8;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = base_idx + i;
        if (idx < n) {
            float log_pred = log_predictions[idx];
            float target = targets[idx];
            thread_sum += expf(log_pred) - target * log_pred;
        }
    }
    
    // First level reduction within each warp using shuffle
    thread_sum = warp_reduce_sum(thread_sum);
    
    // Store warp results in registers
    __shared__ float warp_results[32];  // Only need space for one warp's worth of results
    
    // Each warp's first thread stores the result
    if (lane == 0) {
        warp_results[wid] = thread_sum;
    }
    
    block.sync();
    
    // Final reduction using only the first warp
    if (wid == 0) {
        // Load this thread's value from warp results if it's within bounds
        float final_sum = (lane < warps_per_block) ? warp_results[lane] : 0.0f;
        
        // Final warp-level reduction
        final_sum = warp_reduce_sum(final_sum);
        
        // First thread in block adds to global output
        if (lane == 0) {
            atomicAdd(output, final_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Configure launch parameters for optimal warp usage
    const int threads_per_block = 256;  // 8 warps per block
    const int elements_per_thread = 8;
    const int elements_per_block = threads_per_block * elements_per_thread;
    const int blocks = (n + elements_per_block - 1) / elements_per_block;
    
    kl_div_kernel<<<blocks, threads_per_block>>>(
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