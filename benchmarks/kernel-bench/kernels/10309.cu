#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel_unrolled(const float* x, float* y, int n) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (idx + i * blockDim.x < n) {
            float xi = x[idx + i * blockDim.x];
            float x_cubed = xi * xi * xi;
            float inner = xi + coeff * x_cubed;
            inner *= sqrt_2_over_pi;
            float tanh_val = tanhf(inner);
            y[idx + i * blockDim.x] = 0.5f * xi * (1.0f + tanh_val);
        }
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    int blocks = (n + threads * 4 - 1) / (threads * 4);
    
    gelu_kernel_unrolled<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation");
}