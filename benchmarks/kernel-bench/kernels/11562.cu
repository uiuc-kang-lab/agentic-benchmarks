#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel_min_sync(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {
    
    // Get global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    // Calculate KL divergence for this thread's elements
    #pragma unroll 4
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Store in shared memory
    partial_sums[threadIdx.x] = sum;
    __syncthreads();  // Single sync point after writing to shared memory
    
    // Parallel reduction in shared memory with minimal syncs
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed within a warp)
    if (threadIdx.x < 32) {
        // Unrolled warp reduction
        if (blockDim.x >= 64) partial_sums[threadIdx.x] += partial_sums[threadIdx.x + 32];
        if (blockDim.x >= 32) partial_sums[threadIdx.x] += partial_sums[threadIdx.x + 16];
        if (blockDim.x >= 16) partial_sums[threadIdx.x] += partial_sums[threadIdx.x + 8];
        if (blockDim.x >= 8) partial_sums[threadIdx.x] += partial_sums[threadIdx.x + 4];
        if (blockDim.x >= 4) partial_sums[threadIdx.x] += partial_sums[threadIdx.x + 2];
        if (blockDim.x >= 2) partial_sums[threadIdx.x] += partial_sums[threadIdx.x + 1];
    }
    
    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel_min_sync<<<blocks, threads, shared_mem>>>(
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