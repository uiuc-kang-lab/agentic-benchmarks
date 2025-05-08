#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Inline device function to compute the sigmoid
__device__ inline float compute_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Inline device function to compute the swish activation
__device__ inline float compute_swish(float x) {
    return x * compute_sigmoid(x);
}

// Combined kernel that uses a grid-stride loop for scalability
// and demonstrates warp-level reduction for potential aggregated operations.
__global__ void swish_combined_kernel(const float* __restrict__ x, 
                                        float* __restrict__ y, 
                                        int64_t n) {
    // Full mask for active lanes in a warp
    const unsigned int full_mask = 0xffffffff;
    const int warpSize = 32;

    // Grid-stride loop handles large arrays efficiently
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        // Compute the swish activation in a modular way
        float val = x[idx];
        float result = compute_swish(val);
        y[idx] = result;

        // Optional: demonstrate warp-level reduction of swish values within the warp
        float warp_sum = result;
        // Reduce within the warp using shuffle operations
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(full_mask, warp_sum, offset);
        }
        // Note: warp_sum holds the sum of swish values for the warp for this iteration.
        // This is for demonstration or future aggregated computations and does not affect y.
    }
}

// CUDA forward function that validates input and launches the combined kernel
torch::Tensor swish_combined_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    swish_combined_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_combined_forward, "Optimized Swish activation with grid-stride loops and warp-level primitives (CUDA)");
}
