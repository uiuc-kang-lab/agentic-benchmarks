#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to compute KL divergence with balanced workload distribution
__global__ void workload_balanced_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Ensure all threads have balanced workload
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float lp = log_predictions[i];
        float tar = targets[i];
        // Compute KL divergence
        sum += expf(lp) - tar * lp;
    }

    // Warp-level reduction using shuffle instructions
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Write warp-level results to shared memory
    __shared__ float warpSums[32];
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    if (lane == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();

    // Block-level reduction
    if (threadIdx.x == 0) {
        float blockSum = 0.0f;
        int numWarps = (blockDim.x + 31) >> 5;
        for (int i = 0; i < numWarps; i++) {
            blockSum += warpSums[i];
        }
        atomicAdd(output, blockSum);
    }
}

// Function exposed to PyTorch
torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;  // Adjusted for better workload balancing
    const int blocks = (n + threads - 1) / threads;

    workload_balanced_kl_div_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Workload Balanced KL divergence forward (CUDA)");
}
