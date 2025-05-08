#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Modular device function for computing ELU activation
__device__ inline float compute_elu(float val, float alpha) {
    return (val > 0.0f) ? val : alpha * (expf(val) - 1.0f);
}

// CUDA kernel using the modular device function
__global__ void elu_kernel_modular(const float* x, float* out, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = compute_elu(x[idx], alpha);
    }
}

// CUDA interface function

// The pybind11 module function is named 'forward' as per the example
torch::Tensor elu_cuda_modular(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    elu_kernel_modular<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_modular, "Modularized ELU activation (CUDA)");
}
