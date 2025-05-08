#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Improved kernel with optimized indexing for better performance
__global__ void swish_optimized_indexing_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float val = x[i];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        y[i] = val * sigmoid;
    }
}

torch::Tensor swish_optimized_indexing_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    swish_optimized_indexing_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_optimized_indexing_forward, "Swish activation forward pass with optimized indexing (CUDA)");
}