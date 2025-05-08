#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Constant memory for storing sigmoid coefficients
__constant__ float sigmoid_coeff[2] = {1.0f, 1.0f};

// Modular device function for computing the sigmoid using constant memory
__device__ inline float compute_sigmoid(float x) {
    return sigmoid_coeff[0] / (sigmoid_coeff[1] + expf(-x));
}

// Modular device function for computing the swish activation
__device__ inline float compute_swish(float x) {
    return x * compute_sigmoid(x);
}

// Optimized Swish kernel using grid-stride loop and modular functions
__global__ void swish_constant_memory_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        y[idx] = compute_swish(x[idx]);
    }
}

// CUDA forward function that validates tensor is on device and launches the kernel
torch::Tensor swish_constant_memory_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    swish_constant_memory_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

// Pybind11 binding to expose the CUDA function
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_constant_memory_forward, "Swish activation forward pass with constant memory optimization (CUDA)");
}
