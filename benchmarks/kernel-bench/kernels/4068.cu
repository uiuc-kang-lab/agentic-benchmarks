#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ float elu_compute(float x, float alpha) {
    return (x > 0) ? x : alpha * (expf(x) - 1);
}

__global__ void elu_kernel_modular(const float* x, float* out, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        out[i] = elu_compute(x[i], alpha);
    }
}

torch::Tensor elu_cuda_modular(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();

    // Use 224 threads per block (7 warps) for potentially better occupancy
    // This can reduce register pressure while maintaining good performance
    const int threads = 224;  // 7 warps
    const int blocks = min((n + threads - 1) / threads, 65535);
    
    elu_kernel_modular<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_modular, "ELU activation with modular device functions (CUDA)");
}