#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x);

template <>
__device__ inline float gelu_function<float>(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

template <>
__device__ inline double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x / 1.4142135623730951));
}

template <typename scalar_t>
__global__ void gelu_kernel(const scalar_t* __restrict__ x,
                          scalar_t* __restrict__ y,
                          size_t numel) {
    // Grid-stride loop to handle multiple elements per thread
    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < numel;
        idx += blockDim.x * gridDim.x) {
        // Coalesced memory access pattern
        scalar_t val = x[idx];
        y[idx] = gelu_function<scalar_t>(val);
        // No __syncthreads() needed as operations are independent
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    const int threads = 256;  // Optimal thread count for H100
    const int max_blocks = 65535;
    const int blocks = min(max_blocks, (numel + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda", ([&] {
        gelu_kernel<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                                   output.data_ptr<scalar_t>(),
                                                   numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA)");
}