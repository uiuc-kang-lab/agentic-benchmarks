#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for computing KL divergence element
__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Warp-level reduction using shuffle down
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Main CUDA kernel with warp-level primitives
__global__ void kl_div_kernel_warp_optimized(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / warpSize;
    const unsigned int lane = tid % warpSize;
    int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    // Compute local sum using stride loop
    float sum = 0.0f;
    for (; idx < n; idx += stride) {
        sum += compute_kl_element(log_predictions[idx], targets[idx]);
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    
    // Write reduced sum from each warp into shared memory
    if (lane == 0) {
        partial_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (tid < blockDim.x / warpSize) ? partial_sums[lane] : 0.0f;
        if (lane == 0) {
            for (int i = 1; i < blockDim.x / warpSize; i++) {
                sum += partial_sums[i];
            }
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = (threads / warpSize) * sizeof(float);
    
    kl_div_kernel_warp_optimized<<<blocks, threads, shared_mem>>>(
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
