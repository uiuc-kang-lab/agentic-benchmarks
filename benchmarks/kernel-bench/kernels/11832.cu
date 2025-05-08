#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute KL divergence for a single element
__device__ __forceinline__ float compute_kldiv(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

// Warp-level reduction using shuffle down; no __syncthreads() needed within a warp
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Minimal synchronization kernel: each thread computes its partial sum in a grid-stride loop, 
// then uses warp-level reduction. Only the first thread of each warp issues an atomicAdd.
__global__ void kl_div_kernel_min_sync(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    // Grid-stride loop: process all elements without extra synchronizations
    for (int i = global_tid; i < n; i += stride) {
        float log_val = log_predictions[i];
        float target_val = targets[i];
        local_sum += compute_kldiv(log_val, target_val);
    }

    // Warp-level reduction using shuffle intrinsics
    local_sum = warp_reduce_sum(local_sum);

    // Only warp leaders (lane 0) perform the atomic add to minimize global memory contention
    if ((tid & (warpSize - 1)) == 0) {
        atomicAdd(output, local_sum);
    }
}

// Host wrapper for the kernel
torch::Tensor kl_div_cuda_forward_min_sync(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    int threads = 256;
    int blocks = min((n + threads - 1) / threads, 1024);

    kl_div_kernel_min_sync<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_min_sync, "KL divergence forward with minimal __syncthreads__ (CUDA)");
}
