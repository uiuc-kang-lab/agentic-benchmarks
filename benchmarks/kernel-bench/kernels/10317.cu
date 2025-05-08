#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel that manually unrolls the computation of GELU for four consecutive elements per thread
__global__ void gelu_kernel_manual_unroll(const float* __restrict__ x, float* __restrict__ y, int n) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    // Calculate base index for this thread
    int base = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // Manually unrolled computation for four elements
    if (base < n) {
        float xi = x[base];
        float x_cubed = xi * xi * xi;
        float inner = (xi + coeff * x_cubed) * sqrt_2_over_pi;
        float tanh_val = tanhf(inner);
        y[base] = 0.5f * xi * (1.0f + tanh_val);
    }

    int idx1 = base + blockDim.x;
    if (idx1 < n) {
        float xi = x[idx1];
        float x_cubed = xi * xi * xi;
        float inner = (xi + coeff * x_cubed) * sqrt_2_over_pi;
        float tanh_val = tanhf(inner);
        y[idx1] = 0.5f * xi * (1.0f + tanh_val);
    }

    int idx2 = base + 2 * blockDim.x;
    if (idx2 < n) {
        float xi = x[idx2];
        float x_cubed = xi * xi * xi;
        float inner = (xi + coeff * x_cubed) * sqrt_2_over_pi;
        float tanh_val = tanhf(inner);
        y[idx2] = 0.5f * xi * (1.0f + tanh_val);
    }

    int idx3 = base + 3 * blockDim.x;
    if (idx3 < n) {
        float xi = x[idx3];
        float x_cubed = xi * xi * xi;
        float inner = (xi + coeff * x_cubed) * sqrt_2_over_pi;
        float tanh_val = tanhf(inner);
        y[idx3] = 0.5f * xi * (1.0f + tanh_val);
    }
}

// Host function that launches the manually unrolled GELU kernel
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    int blocks = (n + threads * 4 - 1) / (threads * 4);

    gelu_kernel_manual_unroll<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation with manual loop unrolling");
}
