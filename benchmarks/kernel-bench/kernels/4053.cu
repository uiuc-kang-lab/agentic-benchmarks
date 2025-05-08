#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses a loop-stride design and read-only data caching with __ldg
// to maximize memory throughput on large tensors. The __restrict__ hints help
// the compiler optimize load/store operations. Unlike the shared memory variant,
// we avoid the overhead of copying data into shared memory since each element is
// used only once, and global memory loads benefit from the read-only cache.
__global__ void efficient_elu_kernel(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        // Use __ldg to take advantage of the read-only data cache
        float val = __ldg(&x[i]);
        out[i] = (val > 0) ? val : alpha * (expf(val) - 1.0f);
    }
}

// The launcher function wraps the kernel call, computing an optimal grid configuration
// and ensuring that all tensor inputs are on CUDA and contiguous.
torch::Tensor efficient_elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    efficient_elu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &efficient_elu_cuda, "Efficient ELU activation (CUDA)");
}
