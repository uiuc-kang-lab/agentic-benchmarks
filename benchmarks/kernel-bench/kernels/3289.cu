#include <torch/extension.h>
#include <cmath>

// CUDA kernel with loop unrolling to evenly distribute workloads across threads
__global__ void swish_unroll_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    int i = tid;

    // Unroll loop: each thread processes 4 elements per iteration if possible
    for (; i + 3 * gridSize < n; i += 4 * gridSize) {
        float v0 = x[i];
        float v1 = x[i + gridSize];
        float v2 = x[i + 2 * gridSize];
        float v3 = x[i + 3 * gridSize];

        y[i]               = v0 * (1.0f / (1.0f + expf(-v0)));
        y[i + gridSize]    = v1 * (1.0f / (1.0f + expf(-v1)));
        y[i + 2 * gridSize] = v2 * (1.0f / (1.0f + expf(-v2)));
        y[i + 3 * gridSize] = v3 * (1.0f / (1.0f + expf(-v3)));
    }
    // Process remaining elements
    for (; i < n; i += gridSize) {
        float v = x[i];
        y[i] = v * (1.0f / (1.0f + expf(-v)));
    }
}

// Forward function to launch the CUDA kernel
torch::Tensor swish_unroll_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    swish_unroll_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_unroll_forward, "Swish activation forward pass with loop unrolling (CUDA)");
}
