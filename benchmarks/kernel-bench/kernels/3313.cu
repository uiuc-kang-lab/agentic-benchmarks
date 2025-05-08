#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void ldg_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        // Using __ldg() for read-only cache
        float val = __ldg(&x[i]);
        float sigmoid = __fdividef(1.0f, 1.0f + expf(-val));
        y[i] = val * sigmoid;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    ldg_swish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward with __ldg__ (CUDA)");
}