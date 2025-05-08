#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ float const_log_predictions[1024];

// Device function for computing KL divergence element
__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Optimized warp reduction
__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 32/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void optimized_kl_div_kernel(
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n,
    const bool use_const_mem) {
    
    const unsigned int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    // Main computation loop with memory coalescing
    while (idx < n) {
        float log_pred = use_const_mem ? const_log_predictions[idx] : 
                                       __ldg(&const_log_predictions[idx]); // Use cache operator
        float target = __ldg(&targets[idx]); // Use cache operator
        sum += compute_kl_element(log_pred, target);
        idx += stride;
    }
    
    // Warp-level reduction
    sum = warp_reduce(sum);
    
    // Block-level reduction using shared memory only for inter-warp communication
    if (tid % 32 == 0) partial_sums[tid/32] = sum;
    __syncthreads();
    
    // Final reduction and write result
    if (tid < (blockDim.x/32)) {
        float warp_sum = partial_sums[tid];
        if (tid == 0) {
            float block_sum = 0;
            for (int i = 0; i < blockDim.x/32; ++i)
                block_sum += partial_sums[i];
            atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward_optimized(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = (threads/32) * sizeof(float); // Reduced shared memory usage
    
    bool use_const_mem = (n <= 1024);
    if (use_const_mem) {
        cudaMemcpyToSymbol(const_log_predictions, log_predictions.data_ptr<float>(), 
                          n * sizeof(float));
    }
    
    optimized_kl_div_kernel<<<blocks, threads, shared_mem>>>(
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        use_const_mem
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_optimized, "Optimized KL divergence forward (CUDA)");
}