#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute single KL divergence element
__device__ __forceinline__ float compute_single_kl(
    const float log_pred,
    const float target) {
    return expf(log_pred) - target * log_pred;
}

// Process 4 elements at once using vectorized loads
__device__ __forceinline__ float compute_vectorized_kl(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    const int idx) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sum += compute_single_kl(log_predictions[idx + i], targets[idx + i]);
    }
    return sum;
}

// Warp-level reduction using shuffle
__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
__device__ __forceinline__ void block_reduce(
    float thread_sum,
    float* shared_mem,
    float* output,
    const unsigned int tid) {
    
    shared_mem[tid] = thread_sum;
    __syncthreads();
    
    // Reduce within first warp
    if (tid < 32) {
        float warp_sum = 0.0f;
        #pragma unroll
        for (int i = tid; i < 256; i += 32) {
            warp_sum += shared_mem[i];
        }
        warp_sum = warp_reduce(warp_sum);
        
        if (tid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    extern __shared__ float shared_mem[];
    
    const unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + tid * 4;
    const unsigned int stride = blockDim.x * gridDim.x * 4;
    
    float thread_sum = 0.0f;
    
    // Process chunks of 4 elements per thread
    while (idx + 3 < n) {
        thread_sum += compute_vectorized_kl(log_predictions, targets, idx);
        idx += stride;
    }
    
    // Handle remaining elements
    while (idx < n) {
        thread_sum += compute_single_kl(log_predictions[idx], targets[idx]);
        idx++;
    }
    
    // Perform block-level reduction
    block_reduce(thread_sum, shared_mem, output, tid);
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads * 4 - 1) / (threads * 4), 1024);
    const int shared_mem = threads * sizeof(float);
    
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