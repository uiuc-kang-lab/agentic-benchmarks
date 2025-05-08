#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define constant memory for frequently used data
__constant__ float constant_one = 1.0f;

__global__ void softsign_kernel_constant_memory(const float* x, float* out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        out[idx] = x[idx] / (constant_one + fabsf(x[idx]));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    softsign_kernel_constant_memory<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), num_elements
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with constant memory (CUDA)");}
