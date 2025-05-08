#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel using grid-stride loop to handle large workloads and ensure correct boundary handling
__global__ void leaky_relu_kernel_stride(const float* __restrict__ x, float* __restrict__ out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread using the grid-stride loop
    for (; idx < n; idx += stride) {
        float val = x[idx];
        out[idx] = (val > 0.0f) ? val : val * negative_slope;
    }
}

// Forward function that launches the kernel
torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    // Configure launch parameters
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    leaky_relu_kernel_stride<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        negative_slope,
        n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward with grid-stride loop (CUDA)");
}
