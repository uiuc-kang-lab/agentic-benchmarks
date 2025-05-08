#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Template specializations for GELU function to preserve precision
template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x);

template <>
__device__ inline float gelu_function<float>(float x) {
    // GELU(x) = x * 0.5f * (1 + erf(x / sqrt(2)))
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

template <>
__device__ inline double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x / 1.4142135623730951));
}

// Kernel ensuring memory coalescing by mapping threads to consecutive memory locations
template <typename scalar_t>
__global__ void gelu_coalesced_kernel(const scalar_t* __restrict__ input,
                                       scalar_t* __restrict__ output,
                                       size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    // Each thread processes elements spaced by the stride
    for (size_t i = tid; i < n; i += stride) {
        const scalar_t v = __ldg(&input[i]);
        output[i] = gelu_function(v);
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    auto y = torch::empty_like(x);
    size_t n = x.numel();

    int threads = 256; // using a block size that maps well to GPU cores
    int blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_coalesced_kernel", ([&] {
        gelu_coalesced_kernel<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU forward with coalesced memory accesses (CUDA)");
}
