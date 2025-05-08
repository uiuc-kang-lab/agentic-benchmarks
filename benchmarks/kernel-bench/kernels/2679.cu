#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float leaky_relu_calc(float val, float negative_slope) {
    return val > 0 ? val : val * negative_slope;
}

__global__ void leaky_relu_kernel_unrolled(const float* __restrict__ x, 
                                         float* __restrict__ out, 
                                         float negative_slope, 
                                         int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int unroll_factor = 4;
    
    // Process 4 elements per thread
    for (int i = tid; i < n - (unroll_factor - 1); i += stride * unroll_factor) {
        float val1 = x[i];
        float val2 = x[i + stride];
        float val3 = x[i + stride * 2];
        float val4 = x[i + stride * 3];

        out[i] = leaky_relu_calc(val1, negative_slope);
        out[i + stride] = leaky_relu_calc(val2, negative_slope);
        out[i + stride * 2] = leaky_relu_calc(val3, negative_slope);
        out[i + stride * 3] = leaky_relu_calc(val4, negative_slope);
    }

    // Handle remaining elements
    for (int i = tid + (n / unroll_factor) * unroll_factor; i < n; i += stride) {
        float val = x[i];
        out[i] = leaky_relu_calc(val, negative_slope);
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    leaky_relu_kernel_unrolled<<<blocks, threads>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        negative_slope, 
        n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward unrolled (CUDA)");
}