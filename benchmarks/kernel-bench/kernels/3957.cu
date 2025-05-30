#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_kernel_optimized(const float* x, float* out, int num_elements) {
    const int threads_per_block = 256;
    const int total_threads = blockDim.x * gridDim.x;
    
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < num_elements; 
        idx += total_threads) {
        float val = x[idx];
        out[idx] = val / (1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    const int threads = 256;
    const int blocks = 2048;
    
    softsign_kernel_optimized<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), num_elements);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign optimized with grid-stride loops (CUDA)");
}