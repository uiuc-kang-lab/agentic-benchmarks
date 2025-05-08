#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for KL divergence calculation with warp-level optimizations
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Warp size constant
    constexpr int WARP_SIZE = 32;
    
    // Get global thread ID
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    // Shared memory for partial sums - aligned to warps
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    // Process elements in warp-aligned chunks
    const int stride = blockDim.x * gridDim.x;
    int idx = gid;
    
    // Main loop processes full warps of data
    #pragma unroll 4
    while (idx + WARP_SIZE - 1 < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
        idx += stride;
    }
    
    // Handle remaining elements
    if (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        partial_sums[wid] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < (blockDim.x / WARP_SIZE)) {
        float warp_sum = partial_sums[tid];
        if (tid == 0) {
            for (int i = 1; i < blockDim.x / WARP_SIZE; i++) {
                warp_sum += partial_sums[i];
            }
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters optimized for warp alignment
    const int threads = 256; // Multiple of warp size (32)
    const int blocks = (n + threads - 1) / threads;
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