#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Modular device function for computing the sigmoid
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Modular device function for computing the swish activation
__device__ __forceinline__ float swish(float x) {
    return x * sigmoid(x);
}

// Kernel using grid-stride loop and modular device functions
__global__ void swish_kernel_modular(const float* x, float* y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        y[i] = swish(x[i]);
    }
}

// CUDA forward function
torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;

    swish_kernel_modular<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Modular Swish activation forward pass (CUDA)");
}
