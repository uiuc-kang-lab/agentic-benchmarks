#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define constant memory for alpha
__constant__ float d_alpha;

__global__ void elu_kernel_constant(const float* __restrict__ x, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        float val = x[i];
        out[i] = (val > 0) ? val : d_alpha * (expf(val) - 1);
    }
}

torch::Tensor elu_constant_memory_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    const int n = x.numel();

    // Copy alpha to constant memory
    cudaMemcpyToSymbol(d_alpha, &alpha, sizeof(float));

    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;

    elu_kernel_constant<<<blocks, threads>>>(x.data_ptr<float>(),
                                             out.data_ptr<float>(),
                                             n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_constant_memory_cuda, "ELU activation with constant memory (CUDA)");
}
