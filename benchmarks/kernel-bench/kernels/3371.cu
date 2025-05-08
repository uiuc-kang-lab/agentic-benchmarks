#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized Swish kernel ensuring even workload distribution using grid-stride loop
__global__ void swish_kernel_optimized(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        float v = x[idx];
        float sig = 1.0f / (1.0f + __expf(-v));
        y[idx] = v * sig;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 512;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 144 * 2);  // Limit blocks more conservatively for H100

    swish_kernel_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass (CUDA) with optimized workload distribution");
}
