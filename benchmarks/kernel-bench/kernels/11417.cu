#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__
float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n,
    const int elements_per_thread) {
    
    // Calculate thread indices
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Shared memory for warp results
    extern __shared__ float warp_results[];
    
    float sum = 0.0f;
    
    // Each thread processes multiple elements with balanced distribution
    const int start_idx = global_tid * elements_per_thread;
    const int end_idx = min(start_idx + elements_per_thread, n);
    
    #pragma unroll 4
    for (int idx = start_idx; idx < end_idx; idx++) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Handle remaining elements
    const int total_threads = gridDim.x * blockDim.x;
    const int base_remaining = total_threads * elements_per_thread;
    
    for (int idx = base_remaining + global_tid; idx < n; idx += total_threads) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction
    sum = warp_reduce(sum);
    
    // Store warp results
    if (lane_id == 0) {
        warp_results[warp_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction using first warp
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x / 32)) ? warp_results[lane_id] : 0.0f;
        sum = warp_reduce(sum);
        
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
    
    // Optimize thread and block configuration
    const int threads_per_block = 512; // Increased for better occupancy
    const int warps_per_block = threads_per_block / 32;
    const int shared_mem_size = warps_per_block * sizeof(float);
    
    // Calculate elements per thread for balanced distribution
    const int desired_total_threads = 81920; // Target number for H100
    const int num_blocks = min(128, (desired_total_threads + threads_per_block - 1) / threads_per_block);
    const int total_threads = num_blocks * threads_per_block;
    const int elements_per_thread = (n + total_threads - 1) / total_threads;
    
    kl_div_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        elements_per_thread
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}