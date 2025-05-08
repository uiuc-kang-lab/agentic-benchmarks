#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

__global__ void leaky_relu_kernel_strided(const float* x, float* out, float negative_slope, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements with stride
    for (int i = tid; i < n; i += stride) {
        const float val = x[i];
        out[i] = val > 0 ? val : val * negative_slope;
        
        // Process next elements if within bounds
        if (i + stride < n) {
            const float val2 = x[i + stride];
            out[i + stride] = val2 > 0 ? val2 : val2 * negative_slope;
        }
        
        if (i + 2 * stride < n) {
            const float val3 = x[i + 2 * stride];
            out[i + 2 * stride] = val3 > 0 ? val3 : val3 * negative_slope;
        }
        
        if (i + 3 * stride < n) {
            const float val4 = x[i + 3 * stride];
            out[i + 3 * stride] = val4 > 0 ? val4 : val4 * negative_slope;
        }
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = BLOCK_SIZE;
    const int blocks = (n + (threads * ELEMENTS_PER_THREAD) - 1) / (threads * ELEMENTS_PER_THREAD);

    leaky_relu_kernel_strided<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward with stride loops (CUDA)");
}