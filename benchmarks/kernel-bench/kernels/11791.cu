#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses one warp per block, avoiding shared memory by performing warp-level reduction
__global__ void warp_only_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Each block is a single warp (32 threads) so that we can use shuffle for full reduction
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;

    // Grid-stride loop processing multiple elements per thread
    for (int i = tid; i < n; i += total_threads) {
        float lp = __ldg(log_predictions + i);
        float t  = __ldg(targets + i);
        sum += expf(lp) - t * lp;
    }

    // Warp-level reduction using shuffle intrinsics (works entirely within one warp)
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // The first thread of the warp writes the block's result
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// Host function to launch the kernel
torch::Tensor warp_only_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Launch kernel with one warp (32 threads) per block to avoid shared memory reductions
    const int threads = 32;
    int blocks = (n + threads - 1) / threads;
    // Optionally cap the number of blocks
    blocks = blocks > 4096 ? 4096 : blocks;

    warp_only_kl_kernel<<<blocks, threads>>>(
         log_predictions.data_ptr<float>(),
         targets.data_ptr<float>(),
         output.data_ptr<float>(),
         n);

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_only_kl_forward, "Warp-only KL divergence using warp-level reduction (CUDA)");
}
