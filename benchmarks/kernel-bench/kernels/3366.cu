#include <torch/extension.h>

__global__ void swish_grid_stride(const float* x, float* y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        float val = x[idx];
        float sigmoid = __fdividef(1.0f, 1.0f + __expf(-val));
        y[idx] = val * sigmoid;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 576);  // 4 blocks/SM * 144 SMs on H100

    swish_grid_stride<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish with grid-stride optimization");
}