#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel that fuses grid-stride loop, warp-level reduction using shuffle instructions, 
// and a block-level reduction using shared memory to accumulate the KL divergence.
__global__ void optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Grid-stride loop to compute each thread's partial KL divergence sum
    while (idx < n) {
        float lp = log_predictions[idx];
        float tar = targets[idx];
        // KL divergence: exp(lp) - tar * lp
        sum += expf(lp) - tar * lp;
        idx += blockDim.x * gridDim.x;
    }

    // Perform warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Each warp writes its reduced result into shared memory
    __shared__ float warpSums[32];
    int lane = threadIdx.x & 31;         // Lane index within the warp
    int warpId = threadIdx.x >> 5;         // Warp index within the block
    if (lane == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();

    // The first thread in the block finalizes the block-level reduction
    if (threadIdx.x == 0) {
        float blockSum = 0.0f;
        int numWarps = (blockDim.x + 31) >> 5; // Number of active warps in the block
        for (int i = 0; i < numWarps; i++) {
            blockSum += warpSums[i];
        }
        // Single atomic addition per block to accumulate the total KL divergence
        atomicAdd(output, blockSum);
    }
}

// CUDA function exposed to PyTorch that launches the kernel
torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;

    optimized_kl_div_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Normalize by the number of elements
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Optimized KL divergence forward (CUDA)");
}
