#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses only a single warp per block to perform the reduction,
// eliminating the need for shared memory operations. Each block is allocated
// 32 threads (a full warp), and the reduction is done entirely using __shfl_down_sync.
// The atomicAdd at the end accumulates each block's result into the global output.

__global__ void warp_reduce_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const unsigned int warpSize = 32;
    // Each block is one warp (32 threads)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;

    // Grid-stride loop to cover all elements
    for (int i = idx; i < n; i += stride) {
        float lp = __ldg(&log_predictions[i]);
        float t = __ldg(&targets[i]);
        // Use __expf for fast exponential computation
        sum += __expf(lp) - t * lp;
    }

    // Perform warp-level reduction using shuffle operations
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Only the first thread in the warp writes the block's result to global memory
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// Host function to launch the kernel

torch::Tensor warp_reduce_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Use one warp per block (32 threads per block)
    int block_size = 32;
    // Calculate the number of blocks needed. Increase the number of blocks to maximize occupancy, but cap it to avoid oversubscription.
    int blocks = (n + block_size - 1) / block_size;
    blocks = min(blocks, 1024);

    warp_reduce_kl_div_kernel<<<blocks, block_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_reduce_kl_div_forward, "KLDivLoss with warp-level reduction only (CUDA)");
}
