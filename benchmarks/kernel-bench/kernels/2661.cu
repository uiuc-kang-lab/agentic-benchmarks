#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void leaky_relu_kernel(const float* x, float* out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] > 0 ? x[idx] : x[idx] * negative_slope;
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256; // Optimized block size
    const int blocks = (n + threads - 1) / threads;

    leaky_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA)");
}