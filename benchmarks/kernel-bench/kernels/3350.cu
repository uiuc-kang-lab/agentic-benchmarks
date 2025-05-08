#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized Swish kernel using grid-stride loop for improved workload distribution
__global__ void swish_kernel_optimized(const float* x, float* y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float val = x[i];
        float sig = 1.0f / (1.0f + expf(-val));
        y[i] = val * sig;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 256;
    // Compute number of blocks; grid-stride loop ensures all elements are processed
    int blocks = (n + threads * 4 - 1) / (threads * 4);
    
    swish_kernel_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Optimized Swish activation forward pass (CUDA)");
}
