#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses a grid-stride loop and enforces memory coalescing by using __restrict__ qualifiers
// and the __ldg intrinsic to help the compiler optimize read-only global memory accesses
__global__ void leaky_relu_coalesced_kernel(const float* __restrict__ x, float* __restrict__ out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        // Use __ldg to load from global memory, which may help with caching and coalescing
        float in_val = __ldg(x + i);
        out[i] = (in_val > 0.0f) ? in_val : in_val * negative_slope;
    }
}

torch::Tensor leaky_relu_forward_coalesced(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 1024;
    int blocks = (n + threads - 1) / threads;

    leaky_relu_coalesced_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_coalesced, "LeakyReLU forward coalesced (CUDA)");
}
