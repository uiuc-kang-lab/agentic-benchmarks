#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void coalesced_elu_kernel(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        float val = __ldg(&x[i]);  // Use __ldg for read-only cache
        out[i] = (val > 0) ? val : alpha * (expf(val) - 1);
    }
}

torch::Tensor coalesced_elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    const int n = x.numel();

    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;

    coalesced_elu_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                              out.data_ptr<float>(),
                                              alpha,
                                              n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &coalesced_elu_cuda, "Coalesced ELU activation (CUDA)");
}