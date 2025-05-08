#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized kernel: use __restrict__ and a stride loop to ensure that threads in the warp
// access consecutive memory locations, improving memory coalescing on global memory.
__global__ void elu_kernel_optimized(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float xi = x[i];
        out[i] = (xi > 0.f) ? xi : alpha * (expf(xi) - 1.f);
    }
}

// CUDA wrapper using the optimized kernel
torch::Tensor optimized_elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;

    elu_kernel_optimized<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_elu_cuda, "Optimized ELU activation (CUDA)");
}
