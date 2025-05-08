#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void elu_kernel(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Align thread index to 128-bit boundary (4 floats)
    int aligned_idx = (idx / 4) * 4;
    
    if (aligned_idx < n) {
        // Use __ldg for read-only memory access
        float val = __ldg(&x[idx]);
        out[idx] = (val > 0) ? val : alpha * (expf(val) - 1);
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    // Ensure block size is multiple of warp size for better memory coalescing
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    elu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda, "ELU activation (CUDA)");
}