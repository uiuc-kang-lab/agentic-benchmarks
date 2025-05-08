#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function for computing the sigmoid activation in an inline fashion
__device__ inline float compute_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Device function for computing the swish activation
__device__ inline float compute_swish(float x) {
    return x * compute_sigmoid(x);
}

// Combined kernel using grid-stride loop to efficiently cover all elements
__global__ void combined_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        y[idx] = compute_swish(x[idx]);
    }
}

// Forward function that prepares the CUDA kernel launch
torch::Tensor combined_swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    combined_swish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

// Bindings to expose the CUDA function to PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &combined_swish_forward, "Combined grid-stride loop swish activation forward pass (CUDA)");
}
