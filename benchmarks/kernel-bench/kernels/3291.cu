#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Unified device function for computing the swish activation
__device__ inline float compute_swish(float x) {
    const float sigmoid = 1.0f / (1.0f + expf(-x));
    return x * sigmoid;
}

// Optimized Swish kernel that applies the activation function element-wise
__global__ void optimized_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = compute_swish(x[index]);
    }
}

// CUDA forward function that validates tensor is on device and launches the kernel
torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    optimized_swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

// Pybind11 binding to expose the CUDA function
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Optimized Swish activation forward pass (CUDA)");
}