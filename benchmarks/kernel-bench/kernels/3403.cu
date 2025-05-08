#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Inline GELU function for float precision
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Vectorized CUDA kernel using __ldg for optimized read-only memory access with block-balanced workload
__global__ void gelu_kernel_optimized(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        size_t numel) {
    // Dynamic index calculation to ensure even workload distribution
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    while (idx < numel) {
        // Using __ldg for read-only cache optimization
        float in_val = __ldg(&input[idx]);
        output[idx] = gelu_function(in_val);
        // Increment index by total number of threads in grid for the next iteration
        idx += stride;
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
                "Only float32 is supported for the vectorized __ldg version");

    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    // Launch the kernel with dynamic indexing for balanced workload among threads
    gelu_kernel_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with block balanced workload");
}
