#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__constant__ int BLOCK_SIZE = 256;

__global__ void softsign_kernel_blocktuned(const float* __restrict__ x, 
                                         float* __restrict__ out, 
                                         const int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    while (idx < num_elements) {
        float val = x[idx];
        out[idx] = val / (1.0f + fabsf(val));
        idx += stride;
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int num_elements = x.numel();
    
    // Use optimal block size for H100
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    
    softsign_kernel_blocktuned<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Block-size tuned Softsign activation (CUDA)");
}