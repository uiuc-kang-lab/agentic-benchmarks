#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Branchless LeakyReLU using arithmetic operations to avoid divergent branches
__global__ void leaky_relu_kernel_branchless(const float* __restrict__ x, float* __restrict__ out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float val = x[i];
        float abs_val = fabsf(val);
        // Compute branchless LeakyReLU: when val>=0, out = val; when val<0, out = val * negative_slope
        out[i] = 0.5f * (val + abs_val) + 0.5f * (val - abs_val) * negative_slope;
    }
}

torch::Tensor leaky_relu_forward_branchless(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;
    
    leaky_relu_kernel_branchless<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_branchless, "LeakyReLU forward branchless (CUDA)");
}
