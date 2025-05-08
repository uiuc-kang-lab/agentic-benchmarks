#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for computing KL divergence of a single element
__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Device function for warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Device function for block-level reduction
__device__ __forceinline__ void block_reduce_sum(
    float thread_sum,
    float* shared_mem,
    float* output) {
    
    const int tid = threadIdx.x;
    shared_mem[tid] = thread_sum;
    __syncthreads();

    // Reduce within warps first
    if (tid < 32) {
        float warp_sum = 0.0f;
        #pragma unroll
        for (int i = tid; i < blockDim.x; i += 32) {
            warp_sum += shared_mem[i];
        }
        warp_sum = warp_reduce_sum(warp_sum);
        
        // First thread in block adds to global result
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
    
    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    
    float thread_sum = 0.0f;
    
    // Compute phase - each thread processes its assigned elements
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Process elements with grid stride
    while (idx < n) {
        thread_sum += compute_kl_div(log_predictions[idx], targets[idx]);
        idx += stride;
    }
    
    // Reduction phase
    block_reduce_sum(thread_sum, shared_mem, output);
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
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