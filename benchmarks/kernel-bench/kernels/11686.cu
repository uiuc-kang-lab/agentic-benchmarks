#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void warp_aligned_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    constexpr int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;
    
    float sum = 0.0f;
    
    // Warp-aligned processing with coalesced memory access
    for (int i = warp_id * warp_size; i < n; i += gridDim.x * blockDim.x) {
        int idx = i + lane_id;
        if (idx < n) {
            float log_pred = __ldg(&log_predictions[idx]);
            float target = __ldg(&targets[idx]);
            sum += expf(log_pred) - target * log_pred;
        }
    }

    // Warp-level reduction without divergence
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Single atomic add per warp
    if (lane_id == 0) {
        atomicAdd(output, sum);
    }
}

torch::Tensor warp_aligned_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Optimized for H100 with 32 warps per SM
    const int warps_per_block = 8;
    const int block_size = warps_per_block * 32;  // 256 threads
    const int grid_size = min(256, (n + block_size - 1) / block_size);

    warp_aligned_kl_div_kernel<<<grid_size, block_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_aligned_kl_div_forward, "Warp-aligned KL divergence forward (CUDA)");
}