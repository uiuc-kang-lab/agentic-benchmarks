#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp/warpgroup size constants
constexpr int WARP_SIZE = 32;

__global__ void warp_optimized_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    // Strided loop for coalesced memory access
    for (int idx = tid; idx < n; idx += stride) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Use one warp for final reduction across block
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(output, sum);
    }
}

torch::Tensor warp_optimized_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Launch config optimized for H100's 68 SMs
    const int threads = 256;
    const int blocks = 128;  // 4 warps per block x128 = 512 concurrent warps

    warp_optimized_kl_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_optimized_kl_forward, "Warp-optimized KL divergence (CUDA)");
}
