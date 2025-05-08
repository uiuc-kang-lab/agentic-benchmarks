#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized softsign kernel using a grid-stride loop and __restrict__ pointers
__global__ void softsign_kernel_opt(const float* __restrict__ in, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < num_elements; i += stride) {
        float val = in[i];
        out[i] = val / (1.0f + fabsf(val));
    }
}

// Host function wrapping the kernel
torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    // Configure launch parameters
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    softsign_kernel_opt<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), num_elements);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Softsign activation (CUDA) with grid-stride loop");
}
