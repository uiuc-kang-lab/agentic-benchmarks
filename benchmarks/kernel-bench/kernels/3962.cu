#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_vector_kernel(const float* __restrict__ x,
                                      float* __restrict__ out,
                                      int num_elements) {
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorized processing with grid-stride loop
    while (idx < num_elements) {
        float val = x[idx];
        // Fast math approximation of (abs(x)/(1 + abs(x)))
        out[idx] = __fdividef(val, 1.0f + fabsf(val));
        idx += stride;
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    const int num_elements = x.numel();

    // Optimized launch configuration
    const int threads = 128;
    const int blocks = (num_elements + threads - 1) / threads;

    softsign_vector_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                              out.data_ptr<float>(),
                                              num_elements);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign with vectorized grid-stride");
}