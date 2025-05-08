#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function for computing sigmoid
__device__ __forceinline__ float compute_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Device function for computing swish activation
__device__ __forceinline__ float compute_swish(float x) {
    return x * compute_sigmoid(x);
}

// Kernel utilizing grid-stride loop and modular device functions for clarity
__global__ void modular_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        y[idx] = compute_swish(x[idx]);
    }
}

// Forward function that launches the modular kernel
torch::Tensor modular_swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    modular_swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &modular_swish_forward, "Refactored Modular Swish activation (CUDA)");
}
