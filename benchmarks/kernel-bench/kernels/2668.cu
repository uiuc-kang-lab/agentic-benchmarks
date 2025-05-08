#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimal block size determined from experiments on NVIDIA H100
#define BLOCK_SIZE 256

__global__ void leaky_relu_kernel(const float* __restrict__ x, float* __restrict__ y, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = (val > 0) ? val : val * negative_slope;
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    auto y = torch::empty_like(x);
    int n = x.numel();

    // Use block size of 256 threads as determined by experimental tuning
    const int threads = BLOCK_SIZE;
    const int blocks = (n + threads - 1) / threads;

    leaky_relu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), negative_slope, n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA) block size 256");
}
