#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes the KL divergence using a grid-stride loop and performs intra-warp reduction with shuffle operations.
// To minimize warp divergence, every thread in a warp participates uniformly in the final atomic addition by adding a fraction of the warp's result.

__global__ void branchless_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int warpSize = 32;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;

    // Grid-stride loop: each thread processes multiple elements
    for (int i = global_tid; i < n; i += stride) {
        float log_pred = __ldg(&log_predictions[i]);
        float target = __ldg(&targets[i]);
        sum += __expf(log_pred) - target * log_pred;
    }

    // Intra-warp reduction using shuffle operations to sum values across threads in a warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Broadcast the warp's reduced sum from lane 0 to all lanes in the warp
    float warp_total = __shfl_sync(0xffffffff, sum, 0);

    // Instead of having only a single thread perform the atomic operation, every thread in the warp
    // contributes equally by adding a fraction (1/warpSize) of the warp's total. This branchless approach
    // eliminates divergent conditional logic within the warp.
    float contribution = warp_total / float(warpSize);
    atomicAdd(output, contribution);
}

// Host function that selects block size based on the input size and launches the kernel

torch::Tensor branchless_warp_atomic_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Dynamically select block size based on problem size
    int block_size = 256;
    if (n < 8192) {
        block_size = 128;
    } else if (n > 65536) {
        block_size = 512;
    }

    const int max_blocks = 256;
    int blocks = min(max_blocks, (n + block_size - 1) / block_size);

    branchless_kl_div_kernel<<<blocks, block_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &branchless_warp_atomic_kl_div_forward, "KLDivLoss with branchless warp-level atomicAdd (CUDA)");
}
