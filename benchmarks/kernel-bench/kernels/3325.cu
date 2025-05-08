#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Store the constant value 1.0f in constant memory
__constant__ float c_one = 1.0f;

__global__ void const_memory_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    // Grid-stride loop
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float val = x[i];
        // Use constant memory for the value 1.0f
        float sigmoid = __fdividef(c_one, c_one + expf(-val)); // Using constant memory for 1.0f
        y[i] = val * sigmoid;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    const_memory_swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass using constant memory (CUDA)");
}
