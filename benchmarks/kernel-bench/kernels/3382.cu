#include <torch/extension.h>

__global__ void swish_vectorized(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = idx * 2;
    
    if (pos < n) {
        float val1 = x[pos];
        y[pos] = val1 / (1.0f + __expf(-val1));
    }
    
    if (pos + 1 < n) {
        float val2 = x[pos + 1];
        y[pos + 1] = val2 / (1.0f + __expf(-val2));
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    const int threads = 256;
    const int elements_per_block = threads * 2;
    int blocks = (n + elements_per_block - 1) / elements_per_block;
    blocks = min(blocks, 144 * 8);  // 8 blocks/SM * 144 SMs

    swish_vectorized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish vectorized with 2 elements per thread");
}