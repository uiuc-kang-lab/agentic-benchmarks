#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for KL divergence computation
__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

// Warp reduction function
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel for KL divergence calculation
__global__ void optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Get global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    // Shared memory for partial sums
    extern __shared__ float shared_mem[];
    
    // Local accumulator
    float thread_sum = 0.0f;
    
    // Process elements with grid stride loop
    for (int i = idx; i < n; i += stride) {
        thread_sum += compute_kl_div(log_predictions[i], targets[i]);
    }
    
    // Reduce within warp first
    thread_sum = warp_reduce_sum(thread_sum);
    
    // Write warp results to shared memory
    if (threadIdx.x % 32 == 0) {
        shared_mem[threadIdx.x / 32] = thread_sum;
    }
    __syncthreads();
    
    // Block-level reduction
    if (threadIdx.x < blockDim.x / 32) {
        float block_sum = shared_mem[threadIdx.x];
        block_sum = warp_reduce_sum(block_sum);
        if (threadIdx.x == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = (threads / 32) * sizeof(float);
    
    optimized_kl_div_kernel<<<blocks, threads, shared_mem>>>(
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