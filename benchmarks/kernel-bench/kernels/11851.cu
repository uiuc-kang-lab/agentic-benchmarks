#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes the KL divergence using an optimized reduction strategy.
// Each thread calculates a partial sum over several elements, then a warp-level reduction is performed
// using __shfl_down_sync(). Warp leaders write their result to shared memory,
// and a final reduction is done by the first warp in the block before atomically updating the global sum.

__global__ void kldiv_optimized_reduction_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Loop over elements assigned to this thread
    for (; idx < n; idx += stride) {
        float log_val = log_predictions[idx];
        float target_val = targets[idx];
        sum += expf(log_val) - target_val * log_val;
    }

    // Intra-warp reduction using shuffle
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Each warp's lane 0 writes its sum to shared memory
    int lane = threadIdx.x & (warpSize - 1);
    __shared__ float shared[32];  // Enough for up to 1024 threads (32 warps)
    if (lane == 0) {
        shared[threadIdx.x / warpSize] = sum;
    }
    __syncthreads();

    // Final reduction among warp sums done by the first warp
    if (threadIdx.x < warpSize) {
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        float block_sum = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// CUDA function exposed to PyTorch

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    kldiv_optimized_reduction_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Optimized KL divergence forward using shared memory and warp-level reduction (CUDA)");
}
