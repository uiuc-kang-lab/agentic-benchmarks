#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel_optimized(const float* x, float* y, int n) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float xi = x[i];
        float x_cubed = xi * xi * xi;
        float inner = xi + coeff * x_cubed;
        inner *= sqrt_2_over_pi;
        float tanh_val = tanhf(inner);
        y[i] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_forward_optimized(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    gelu_kernel_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_optimized", &gelu_forward_optimized, "GELU forward optimized CUDA implementation");
}