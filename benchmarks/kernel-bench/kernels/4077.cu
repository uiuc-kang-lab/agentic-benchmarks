#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel using stride loops to handle workloads larger than the number of threads
__global__ void elu_stride_kernel(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        float val = x[i];
        out[i] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
    }
}

// CUDA wrapper using the stride-loop kernel
torch::Tensor stride_loop_elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256; // Block dimension
    const int blocks = (n + threads - 1) / threads;

    elu_stride_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &stride_loop_elu_cuda, "Stride loop ELU activation (CUDA)");
}
