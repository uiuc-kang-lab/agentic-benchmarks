#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void elu_kernel_stride(const float* x, float* out, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i = idx;

// Unroll loop by processing two elements per iteration
while (i + stride < n) {
    float v0 = x[i];
    float v1 = x[i + stride];
    out[i] = (v0 > 0) ? v0 : alpha * (expf(v0) - 1);
    out[i + stride] = (v1 > 0) ? v1 : alpha * (expf(v1) - 1);
    i += 2 * stride;
}
if (i < n) {
    float v = x[i];
    out[i] = (v > 0) ? v : alpha * (expf(v) - 1);
}
}

torch::Tensor elu_cuda_stride(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    elu_kernel_stride<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_stride, "ELU activation with stride loop (CUDA)");
}