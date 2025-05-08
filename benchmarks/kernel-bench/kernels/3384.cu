#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Modular device function for GELU computation
__device__ inline float gelu_function_modular(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

__device__ inline double gelu_function_modular(double x) {
    return x * 0.5 * (1.0 + erf(x / 1.4142135623730951));
}

// CUDA kernel that applies the GELU activation element-wise using modular functions.
template <typename scalar_t>
__global__ void gelu_kernel_modular(const scalar_t* __restrict__ x,
                                    scalar_t* __restrict__ y,
                                    size_t numel) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        scalar_t val = x[index];
        y[index] = gelu_function_modular(val);
    }
}

// Forward function callable from Python.
torch::Tensor forward_modular(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    int threads = 1024;
    int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda_modular", ([&] {
        gelu_kernel_modular<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                                           output.data_ptr<scalar_t>(),
                                                           numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_modular", &forward_modular, "GELU activation forward modular (CUDA)");
}