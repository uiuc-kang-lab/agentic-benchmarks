#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel using a stride loop to process elements beyond the number of available threads.
// Each thread processes multiple elements by incrementing its index by the total thread count.
__global__ void elu_kernel_stride(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float val = x[i];
        out[i] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
    }
}

// CUDA wrapper that launches the kernel using a stride loop and verifies boundary handling
// for workloads larger than the number of available threads.
torch::Tensor elu_cuda_stride(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;

    elu_kernel_stride<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_stride, "Stride loop ELU activation (CUDA)");
}
