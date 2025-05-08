#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// GELU function for float precision
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Kernel using stride loops for efficient processing
__global__ void gelu_kernel_stride(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < numel; i += stride) {
        float val = __ldg(&input[i]);
        output[i] = gelu_function(val);
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
                "Only float32 is supported for this optimized version");

    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    // Launch kernel with stride loop
    gelu_kernel_stride<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with stride loop optimization");
}
