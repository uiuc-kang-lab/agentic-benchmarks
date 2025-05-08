#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

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

// CUDA kernel that applies the GELU activation element-wise with minimized warp divergence.
template <typename scalar_t>
__global__ void gelu_kernel_no_divergence(const scalar_t* __restrict__ x,
                                          scalar_t* __restrict__ y,
                                          size_t numel) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t val = 0;
    if (index < numel) {
        val = x[index];
    }
    scalar_t result = gelu_function<scalar_t>(val);
    if (index < numel) {
        y[index] = result;
    }
}

// Forward function callable from Python.
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda_no_divergence", ([&] {
        gelu_kernel_no_divergence<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
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