#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function for computing the swish activation
__device__ inline float compute_swish(float x) {
    return x / (1.0f + expf(-x));
}

// Optimized Swish kernel using stride loop for efficient workload handling
__global__ void swish_stride_loop_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (; idx < n; idx += stride) {
        y[idx] = compute_swish(x[idx]);
    }
}

// CUDA forward function that validates tensor is on device and launches the kernel
torch::Tensor swish_stride_loop_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;
    
    swish_stride_loop_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

// Pybind11 binding to expose the CUDA function
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_stride_loop_forward, "Swish activation forward pass with stride loop optimization (CUDA)");
}
