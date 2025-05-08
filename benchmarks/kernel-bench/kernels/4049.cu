#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This branchless kernel minimizes warp divergence by using arithmetic operations to
// select the output value, removing conditional branches within each warp.
__global__ void elu_kernel_branchless(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride) {
        // Use __ldg for cached global memory load
        float val = __ldg(&x[i]);
        // Convert boolean (val > 0) to float: 1.0f if true, 0.0f if false
        float pos = (float)(val > 0);
        float neg = 1.0f - pos;
        // Compute both possible outcomes without branching
        float branchless_result = pos * val + neg * (alpha * (expf(val) - 1.0f));
        out[i] = branchless_result;
    }
}

torch::Tensor elu_cuda_branchless(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();
    const int threads = 256; // Align to warp size (32)
    const int blocks = (n + threads - 1) / threads;
    elu_kernel_branchless<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_branchless, "Branchless ELU activation (CUDA)");
}
