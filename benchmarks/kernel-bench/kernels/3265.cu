#include <torch/extension.h>

// Optimized kernel using a grid-stride loop to process multiple elements per thread.
// No atomic operations are used as each thread works on independent output elements.

__global__ void optimized_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        float val = x[idx];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        y[idx] = val * sigmoid;
    }
}

torch::Tensor optimized_swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    optimized_swish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_swish_forward, "Optimized 25_Swish activation forward pass (CUDA)");
}
