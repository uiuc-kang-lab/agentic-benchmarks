#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute single element KL divergence
__device__ __forceinline__ float compute_element_kl(
    const float log_pred,
    const float target) {
    return expf(log_pred) - target * log_pred;
}

// Load data with cache hint
__device__ __forceinline__ void load_data(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float& log_pred,
    float& target,
    const int idx) {
    __ldg(&log_predictions[idx]); // Cache hint for read-only data
    log_pred = log_predictions[idx];
    target = targets[idx];
}

// Warp-level reduction with improved shuffle operations
__device__ __forceinline__ float warp_reduce_kl(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Block-level reduction with shared memory
__device__ __forceinline__ float block_reduce_kl(
    float thread_val,
    float* shared_mem,
    const unsigned int tid,
    const unsigned int lane_id) {
    
    // First warp-level reduction
    thread_val = warp_reduce_kl(thread_val);
    
    // Write reduced warp values to shared memory
    if (lane_id == 0) {
        shared_mem[tid >> 5] = thread_val;
    }
    __syncthreads();
    
    // Final reduction with first warp
    if (tid < 32) {
        thread_val = (tid < (blockDim.x >> 5)) ? shared_mem[tid] : 0.0f;
        thread_val = warp_reduce_kl(thread_val);
    }
    
    return thread_val;
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n,
    const int iterations_per_block) {
    
    extern __shared__ float shared_mem[];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid & 31;
    const unsigned int global_thread_id = blockIdx.x * blockDim.x + tid;
    
    // Persistent thread approach
    for (int iter = 0; iter < iterations_per_block; iter++) {
        float thread_sum = 0.0f;
        
        // Process elements with current thread
        for (int i = global_thread_id + iter * gridDim.x * blockDim.x; 
             i < n; 
             i += gridDim.x * blockDim.x * iterations_per_block) {
            
            float log_pred, target;
            load_data(log_predictions, targets, log_pred, target, i);
            thread_sum += compute_element_kl(log_pred, target);
        }
        
        // Perform block-level reduction
        thread_sum = block_reduce_kl(thread_sum, shared_mem, tid, lane_id);
        
        // Add block result to global output
        if (tid == 0) {
            atomicAdd(output, thread_sum);
        }
        __syncthreads();
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int max_blocks = 1024;
    const int iterations_per_block = 4;
    const int blocks = min((n + threads - 1) / threads, max_blocks);
    const int warps_per_block = threads >> 5;
    const int shared_mem = warps_per_block * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        iterations_per_block
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}