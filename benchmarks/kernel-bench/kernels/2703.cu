#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// LeakyReLU kernel using a grid-stride loop
__global__ void leaky_relu_kernel_stride(const float* x, float* out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        const float val = x[i];
        out[i] = val * ((val > 0.0f) ? 1.0f : negative_slope);
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 1024;
    int blocks = (n + threads - 1) / threads;

    // Cap blocks to device's maximum allowed grid dimension
    int maxBlocks;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    maxBlocks = prop.maxGridSize[0];
    if (blocks > maxBlocks) {
        blocks = maxBlocks;
    }

    leaky_relu_kernel_stride<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA)");
}
