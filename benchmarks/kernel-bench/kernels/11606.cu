#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    
    extern __shared__ float smem[];
    
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int warps_per_block = blockDim.x / 32;
    
    // Each thread processes 4 elements initially
    int idx = blockIdx.x * (blockDim.x * 4) + tid;
    const int stride = blockDim.x * gridDim.x;
    
    float thread_sum = 0.0f;
    
    // Vector loading and processing
    #pragma unroll 4
    while (idx + 3 * stride < n) {
        // Process 4 elements per iteration
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int curr_idx = idx + i * stride;
            if (curr_idx < n) {
                const float log_pred = log_predictions[curr_idx];
                const float target = targets[curr_idx];
                thread_sum += expf(log_pred) - target * log_pred;
            }
        }
        idx += 4 * stride;
    }
    
    // Handle remaining elements
    while (idx < n) {
        const float log_pred = log_predictions[idx];
        const float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
        idx += stride;
    }
    
    // Warp-level reduction first
    thread_sum = warp_reduce_sum(thread_sum);
    
    // Store warp results in shared memory
    if (lane_id == 0) {
        smem[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < warps_per_block) {
        float warp_sum = smem[tid];
        
        // Final warp does the last reduction
        if (warp_id == 0) {
            // Reduce all warp sums
            if (tid < warps_per_block) {
                warp_sum = warp_reduce_sum(warp_sum);
                
                // First thread writes result
                if (lane_id == 0) {
                    atomicAdd(output, warp_sum);
                }
            }
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize block size for warp-alignment
    const int threads = 128; // 4 warps per block
    const int blocks = min((n + threads * 4 - 1) / (threads * 4), 1024);
    const int warps_per_block = threads / 32;
    const int shared_mem = warps_per_block * sizeof(float);
    
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