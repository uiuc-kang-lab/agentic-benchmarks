#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void elu_kernel_optimized(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int pos = idx + i * blockDim.x;
        if (pos < n) {
            float val = x[pos];
            out[pos] = (val > 0) ? val : alpha * (expf(val) - 1);
        }
    }
}

torch::Tensor elu_cuda_optimized(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 512;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);

    elu_kernel_optimized<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_optimized, "Optimized ELU without atomics (CUDA)");
}