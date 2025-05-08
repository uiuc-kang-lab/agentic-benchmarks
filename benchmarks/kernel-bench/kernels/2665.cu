#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void leaky_relu_kernel_optimized(const float* __restrict__ x, 
                                          float* __restrict__ out,
                                          float negative_slope, 
                                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread when possible
    #pragma unroll
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        float val = x[i];
        // Use arithmetic instead of branching
        float pos_mask = (val > 0);
        float neg_mask = 1.0f - pos_mask;
        out[i] = pos_mask * val + neg_mask * (val * negative_slope);
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();

    // Optimize block and grid size for better occupancy
    const int threads = 256;
    const int blocks = min(65535, (n + threads - 1) / threads);

    leaky_relu_kernel_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        negative_slope,
        n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA)");
}