#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Optimized device function for computing the sigmoid
__device__ inline float compute_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Optimized kernel utilizing warp-level primitives for swish activation
__global__ void optimized_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    const unsigned int full_mask = 0xffffffff;
    const int warpSize = 32;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we do not exceed bounds
    if (idx < n) {
        float val = x[idx];
        float swish_val = val * compute_sigmoid(val);
        y[idx] = swish_val;

        // Warp-level reductions can be used optionally here
        float warp_sum = swish_val;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(full_mask, warp_sum, offset);
        }
        // Lane 0 holds the reduction result, if used for further operations.
    }
}

// CUDA forward function for optimizing swish computation
torch::Tensor optimized_swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    optimized_swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_swish_forward, "Optimized Swish activation forward pass with optional warp-level reductions (CUDA)");
}