#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that uses a grid-stride loop to distribute workloads evenly across threads and blocks
__global__ void leaky_relu_kernel_grid(const float* __restrict__ x, float* __restrict__ y, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        float val = x[i];
        y[i] = (val > 0.0f) ? val : val * negative_slope;
    }
}

// Forward function that launches the grid-stride kernel
torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    auto y = torch::empty_like(x);
    int n = x.numel();

    // Use 1024 threads per block for high occupancy
    const int threads = 1024;
    // Use a sufficient number of blocks to cover the tensor
    const int blocks = (n + threads - 1) / threads;

    leaky_relu_kernel_grid<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), negative_slope, n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward with grid-stride loop (CUDA)");
}
