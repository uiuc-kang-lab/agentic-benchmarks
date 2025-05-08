#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void leaky_relu_kernel_nosync(const float* __restrict__ x, 
                                       float* __restrict__ out,
                                       const float negative_slope,
                                       const int n) {
    // Grid-stride loop for better occupancy
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        const float val = x[idx];
        out[idx] = val > 0 ? val : val * negative_slope;
    }
}

torch::Tensor leaky_relu_forward_nosync(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int n = x.numel();
    
    const int threads = 256;  // Optimize thread count for H100, aligned to warp size
    const int blocks = min(65535, (n + threads - 1) / threads);
    
    leaky_relu_kernel_nosync<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        negative_slope,
        n
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_nosync, "LeakyReLU forward nosync (CUDA)");
}