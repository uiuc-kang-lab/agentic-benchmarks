#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel computes hinge loss: output[i] = max(0, 1 - predictions[i]*targets[i])
__global__ void hinge_loss_optimal_kernel(const float* __restrict__ predictions,
                                           const float* __restrict__ targets,
                                           float* __restrict__ output,
                                           const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        // Use __ldg for efficient read-only caching
        float pred = __ldg(&predictions[idx]);
        float targ = __ldg(&targets[idx]);
        output[idx] = fmaxf(0.0f, 1.0f - pred * targ);
    }
}

// Forward function: selects optimal block size dynamically by using cudaOccupancyMaxPotentialBlockSize
// This API experiments with candidate block sizes (32, 64, 128, 256, 512, etc.) and returns the best one
// based on occupancy analysis for the current GPU (e.g., NVIDIA H100). This should help reduce runtime overhead
// and improve kernel performance while returning the correct result.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    const int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    // Determine optimal block size using occupancy API
    int minGridSize;
    int optimalBlockSize;
    cudaError_t occupancyErr = cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &optimalBlockSize, hinge_loss_optimal_kernel, 0, n);
    if (occupancyErr != cudaSuccess) {
        optimalBlockSize = 256;  // fallback to a common configuration if the occupancy query fails
    }
    
    int numBlocks = (n + optimalBlockSize - 1) / optimalBlockSize;

    hinge_loss_optimal_kernel<<<numBlocks, optimalBlockSize>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tuned Hinge Loss Forward with Dynamic Block Size");
}
