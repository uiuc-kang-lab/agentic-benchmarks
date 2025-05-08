#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory buffers for frequently accessed data
__constant__ float const_log_predictions[8192];  // 32KB constant memory limit
__constant__ float const_targets[8192];

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kl_div_kernel_constant(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n,
    const bool use_const_mem) {
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    
    float thread_sum = 0.0f;
    
    // Use constant memory for initial portion if data fits
    if (use_const_mem && gid < 8192) {
        thread_sum += __expf(const_log_predictions[gid]) - const_targets[gid] * const_log_predictions[gid];
        gid += stride;
    }
    
    // Process remaining elements from global memory
    for (; gid < n; gid += stride) {
        float log_pred = log_predictions[gid];
        float target = targets[gid];
        thread_sum += __expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);
    
    // Block-level reduction using shared memory
    __shared__ float warp_results[8];  // For 256 threads = 8 warps
    int warp_id = tid / 32;
    
    if (tid % 32 == 0) {
        warp_results[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0 && tid < 8) {
        float final_sum = warp_results[tid];
        final_sum = warp_reduce_sum(final_sum);
        
        if (tid == 0) {
            atomicAdd(output, final_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Copy initial portion to constant memory if size allows
    bool use_const_mem = false;
    if (n <= 8192) {
        cudaMemcpyToSymbol(const_log_predictions, 
                          log_predictions.data_ptr<float>(), 
                          n * sizeof(float));
        cudaMemcpyToSymbol(const_targets,
                          targets.data_ptr<float>(),
                          n * sizeof(float));
        use_const_mem = true;
    }
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    
    kl_div_kernel_constant<<<blocks, threads, 8 * sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        use_const_mem
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward with constant memory (CUDA)");
}