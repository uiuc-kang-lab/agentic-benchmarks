#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float compute_inner(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    return (x + coeff * x_cubed) * sqrt_2_over_pi;
}

__device__ __forceinline__ float compute_gelu(float x) {
    float inner = compute_inner(x);
    float tanh_val = __tanhf(inner);
    return 0.5f * x * (1.0f + tanh_val);
}

__global__ void gelu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = compute_gelu(x[i]);
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    gelu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation");
}