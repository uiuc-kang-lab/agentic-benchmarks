#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using a grid-stride loop for handling workloads larger than available threads
__global__ void stride_loop_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Compute global thread index and stride
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;

    // Use a grid-stride loop to process all elements
    for (int i = idx; i < n; i += stride) {
        // Ensure correct boundary handling
        float lp = log_predictions[i];
        float tar = targets[i];
        // KL divergence term: exp(lp) - tar * lp
        sum += expf(lp) - tar * lp;
    }

    // Warp-level reduction using shuffle
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Each warp writes its partial sum to shared memory
    __shared__ float sharedSum[32];
    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x >> 5;
    if (lane == 0) {
        sharedSum[warpId] = sum;
    }
    __syncthreads();

    // Final reduction by the first thread in the block
    if (threadIdx.x == 0) {
        float blockSum = 0.0f;
        int numWarps = (blockDim.x + 31) / 32;
        for (int i = 0; i < numWarps; i++) {
            blockSum += sharedSum[i];
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

    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;

    stride_loop_kl_div_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward using stride loop (CUDA)");
}
