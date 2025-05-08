#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel minimizes warp divergence by computing the LeakyReLU operation in a branchless manner.
// It uses fabsf to get the absolute value and then computes the activation as a combination of x and fabsf(x).

__global__ void leaky_relu_kernel(const float* x, float* out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float abs_val = fabsf(val);
        // Branchless computation: for positive x, (x + fabs(x)) = 2x and (x - fabs(x)) = 0; for negative, it's 0 and 2x.
        out[idx] = 0.5f * (val + abs_val) + 0.5f * (val - abs_val) * negative_slope;
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;

    leaky_relu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "Branchless LeakyReLU forward (CUDA)");
}
