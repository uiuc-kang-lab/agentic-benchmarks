#include <torch/extension.h>

__constant__ float ONE_HALF = 1.0f;

__global__ void swish_constant_mem(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        const float val = x[idx];
        const float sigmoid = ONE_HALF / (ONE_HALF + expf(-val));
        y[idx] = val * sigmoid;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 576);  // 4 blocks/SM * 144 SMs

    swish_constant_mem<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish with constant memory");
}
