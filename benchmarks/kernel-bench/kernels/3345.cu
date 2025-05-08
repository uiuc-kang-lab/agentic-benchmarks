#include <torch/extension.h>

__global__ void swish_kernel(const float* x, float* y, int64_t n) {
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const float val = x[index];
        const float sigmoid = 1.0f / (1.0f + expf(-val));
        y[index] = val * sigmoid;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    swish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass (CUDA)");
}