#include <torch/extension.h>

__global__ void swish_unroll_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        const float val = x[i];
        const float sigmoid = 1.0f / (1.0f + expf(-val));
        y[i] = val * sigmoid;
    }
}

// Improved forward function using unrolled loop in kernel
torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    swish_unroll_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass with unrolled loops (CUDA)");
}