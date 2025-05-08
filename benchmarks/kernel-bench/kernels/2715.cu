#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device function for LeakyReLU computation
__device__ inline float leaky_relu(float x, float negative_slope) {
    return x > 0.0f ? x : x * negative_slope;
}

// Kernel function using a grid-stride loop
__global__ void leaky_relu_kernel_tunable(const float* __restrict__ x, float* __restrict__ out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = leaky_relu(x[i], negative_slope);
    }
}

// Host function with tunable block size parameter
// Experiments with block sizes such as 32, 64, 128, 256, 512 to find the optimal configuration
torch::Tensor leaky_relu_forward_tunable(torch::Tensor x, float negative_slope, int block_size) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();
    int blocks = (n + block_size - 1) / block_size;
    
    leaky_relu_kernel_tunable<<<blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_tunable, "LeakyReLU forward tunable (CUDA)");
}
