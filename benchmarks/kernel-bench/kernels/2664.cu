#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void leaky_relu_kernel_vectorized(const float4* x, float4* out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elements_per_thread = 4;
    if (idx * elements_per_thread < n) {
        float4 input = x[idx];
        
        // Process all 4 elements in parallel
        float4 result;
        result.x = input.x > 0 ? input.x : input.x * negative_slope;
        result.y = input.y > 0 ? input.y : input.y * negative_slope;
        result.z = input.z > 0 ? input.z : input.z * negative_slope;
        result.w = input.w > 0 ? input.w : input.w * negative_slope;
        
        out[idx] = result;
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();
    int vector_size = 4;
    int n_vectors = (n + vector_size - 1) / vector_size;

    const int threads = 256;
    const int blocks = (n_vectors + threads - 1) / threads;

    leaky_relu_kernel_vectorized<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        negative_slope,
        n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA)");
}