#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Experiment with block sizes: try 32, 64, 128, 256, 512 (here we use 256 as a starting point)
constexpr int BLOCK_SIZE = 256;

__global__ void leaky_relu_kernel(const float* __restrict__ x, float* __restrict__ out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        float val = x[idx];
        out[idx] = (val > 0.0f) ? val : val * negative_slope;
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    leaky_relu_kernel<<<blocks, BLOCK_SIZE>>>(x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA) with block size tuning");
}
