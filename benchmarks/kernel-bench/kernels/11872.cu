#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to compute KL divergence with balanced workload distribution
__global__ void balanced_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Use grid-stride loop to ensure balanced workload
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float lp = log_predictions[i];
        float tar = targets[i];
        sum += expf(lp) - tar * lp;
    }

    // Warp-level reduction using shuffle instructions
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Shared memory for block-level reduction
    __shared__ float warpSums[32];
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    if (lane == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();

    // Final reduction by the first thread in the block
    if (threadIdx.x == 0) {
        float blockSum = 0.0f;
        int numWarps = (blockDim.x + 31) >> 5;
        for (int i = 0; i < numWarps; i++) {
            blockSum += warpSums[i];
        }
        atomicAdd(output, blockSum);
    }
}

// CUDA function exposed to PyTorch
torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;  // Adjusted for better balance
    const int blocks = (n + threads - 1) / threads;

    balanced_kl_div_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Balanced KL divergence forward (CUDA)");
}