#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Declare constant memory for negative slope
__constant__ float d_negative_slope;

__global__ void leaky_relu_kernel_constant(const float* __restrict__ x, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = x[idx];
        // Use the constant memory version of negative_slope
        out[idx] = val > 0 ? val : val * d_negative_slope;
    }
}

torch::Tensor leaky_relu_forward_constant(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();

    // Copy negative_slope to constant memory
    cudaMemcpyToSymbol(d_negative_slope, &negative_slope, sizeof(float));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    leaky_relu_kernel_constant<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_constant, "LeakyReLU forward with constant memory (CUDA)");
}