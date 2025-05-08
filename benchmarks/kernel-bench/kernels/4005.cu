#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Store alpha in constant memory for fast access
__constant__ float const_alpha;

__global__ void elu_kernel_constant(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        out[idx] = (val > 0) ? val : const_alpha * (expf(val) - 1);
    }
}

// Host interface function
// It sets the constant memory value and launches the kernel.
torch::Tensor elu_cuda_constant(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    // Copy alpha to constant memory
    cudaMemcpyToSymbol(const_alpha, &alpha, sizeof(float));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    elu_kernel_constant<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_constant, "ELU activation using constant memory (CUDA)");
}
