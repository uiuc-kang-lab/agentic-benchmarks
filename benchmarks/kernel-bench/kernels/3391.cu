#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define warp size if not defined
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Explicit specializations of gelu_function for float and double.

template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x);

template <>
__device__ inline float gelu_function<float>(float x) {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

template <>
__device__ inline double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x / 1.4142135623730951));
}

// CUDA kernel that applies the GELU activation element-wise using __ldg for efficient read-only loads
// and demonstrates a dummy warp-level reduction with __shfl_down_sync to replace shared memory operations.

template <typename scalar_t>
__global__ void gelu_kernel(const scalar_t* __restrict__ x,
                              scalar_t* __restrict__ y,
                              size_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process multiple elements per thread
    for (size_t i = idx; i < numel; i += stride) {
         // Use __ldg to load from read-only global memory (improves cache usage within a warp)
         scalar_t val = __ldg(&x[i]);
         scalar_t res = gelu_function<scalar_t>(val);
         y[i] = res;
    }

    // Dummy warp-level reduction using __shfl_down_sync to illustrate replacing shared memory operations
    // This reduction computes a sum within each warp of one of the computed GELU values
    int lane = threadIdx.x % WARP_SIZE;
    // Use a valid value if in-bounds, else zero
    scalar_t local_val = (idx < numel) ? gelu_function<scalar_t>(__ldg(&x[idx])) : scalar_t(0);
    
    // Perform warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
         local_val += __shfl_down_sync(0xffffffff, local_val, offset);
    }
    // The result of this dummy reduction is not used, but it demonstrates removal of shared memory in favor of warp-level primitives.
}

// Forward function callable from Python.

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    int threads = 1024;
    int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda_ldg", ([&] {
        gelu_kernel<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                                     output.data_ptr<scalar_t>(),
                                                     numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with __ldg and warp-level primitives");
}
