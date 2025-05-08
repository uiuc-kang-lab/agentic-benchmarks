#include <torch/extension.h>

__global__ void optimized_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        const float val = x[i];
        const float sigmoid = __frcp_rn(1.0f + expf(-val));  // Use reciprocal function to improve performance
        y[i] = val * sigmoid;
    }
}

torch::Tensor optimized_swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    optimized_swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_swish_forward, "Optimized Swish activation forward pass (CUDA)");
}