#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kldiv_minimal_atomics_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Each thread computes partial sum
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
        idx += blockDim.x * gridDim.x;
    }

    // Warp-level reduction using shuffle
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(mask, sum, offset);

    // Block-level reduction in shared memory
    __shared__ float block_sum[32];
    if (threadIdx.x % 32 == 0)
        block_sum[threadIdx.x / 32] = sum;
    __syncthreads();

    // Final block reduction by first warp
    if (threadIdx.x < 32) {
        float val = (threadIdx.x < blockDim.x / 32) ? block_sum[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(mask, val, offset);

        // Single atomic add per block
        if (threadIdx.x == 0)
            atomicAdd(output, val);
    }
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;

    kldiv_minimal_atomics_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward with minimal atomics");
}