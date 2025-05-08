#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device function for LeakyReLU computation
__device__ float leaky_relu_device(float x, float negative_slope) {
    return x > 0 ? x : x * negative_slope;
}

// Kernel function
__global__ void leaky_relu_kernel_streams(const float* x, float* out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = leaky_relu_device(x[idx], negative_slope);
    }
}

// Host function
torch::Tensor leaky_relu_forward_streams(torch::Tensor x, float negative_slope, cudaStream_t stream) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;

    leaky_relu_kernel_streams<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", [](torch::Tensor x, float negative_slope) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        auto result = leaky_relu_forward_streams(x, negative_slope, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        return result;
    }, "LeakyReLU forward with streams (CUDA)");
}