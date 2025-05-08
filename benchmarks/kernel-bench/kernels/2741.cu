#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void leaky_relu_unrolled_kernel(const float* x, float* out, float negative_slope, int n) {
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    #pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        int idx = base_idx + i;
        if (idx < n) {
            out[idx] = fmaxf(x[idx], x[idx] * negative_slope);
        }
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 1024;
    const int unroll_factor = 4;
    const int blocks = (n + threads * unroll_factor - 1) / (threads * unroll_factor);

    leaky_relu_unrolled_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward with unrolled loops (CUDA)");
}