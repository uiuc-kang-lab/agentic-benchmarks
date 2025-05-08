#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses __ldg() for read-only global memory loads and processes data in 128-bit chunks
// (i.e., using float4) assuming the input tensor is 128-bit aligned. Tail elements are handled separately.

__global__ void leaky_relu_kernel_vectorized(const float* __restrict__ x, float* __restrict__ out, float negative_slope, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process as many groups of 4 floats as possible (128-bit loads)
    int num4 = n / 4;  // number of float4 groups
    for (int i = tid; i < num4; i += stride) {
        // use __ldg for read-only load from global memory
        float4 in_val = __ldg(((const float4*) x) + i);
        float4 res;
        res.x = (in_val.x > 0.f) ? in_val.x : in_val.x * negative_slope;
        res.y = (in_val.y > 0.f) ? in_val.y : in_val.y * negative_slope;
        res.z = (in_val.z > 0.f) ? in_val.z : in_val.z * negative_slope;
        res.w = (in_val.w > 0.f) ? in_val.w : in_val.w * negative_slope;
        ((float4*) out)[i] = res;
    }

    // Handle remaining elements if n is not a multiple of 4
    int offset = num4 * 4;
    for (int i = tid + offset; i < n; i += stride) {
        float val = __ldg(&x[i]);
        out[i] = (val > 0.f) ? val : val * negative_slope;
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    // Use a block size that typically provides good occupancy; adjust as needed
    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;

    leaky_relu_kernel_vectorized<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward optimized with __ldg and vectorized loads (CUDA)");
}
