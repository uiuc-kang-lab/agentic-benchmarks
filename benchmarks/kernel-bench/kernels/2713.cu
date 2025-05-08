#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void leaky_relu_kernel_even_distribution(const float* __restrict__ x, float* out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = x[i] > 0 ? x[i] : x[i] * negative_slope;
    }
}

torch::Tensor leaky_relu_forward_even_distribution(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 1024; // Number of threads per block, should be a multiple of 32
    int blocks = (n + threads - 1) / threads;

    // Ensure the number of blocks does not exceed the maximum grid size
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    blocks = min(blocks, prop.maxGridSize[0]);

    leaky_relu_kernel_even_distribution<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_even_distribution, "LeakyReLU forward even distribution (CUDA)");
}