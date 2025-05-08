#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device GELU function specializations
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

// CUDA kernel with manual loop unrolling (factor 4) to reduce loop overhead
template <typename scalar_t>
__global__ void gelu_kernel_unroll(const scalar_t* __restrict__ x,
                                   scalar_t* __restrict__ y,
                                   size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t i = idx;
    
    // Unrolled loop: process 4 elements per iteration
    while (i + 3 * stride < numel) {
        #pragma unroll
        {
            y[i]             = gelu_function<scalar_t>(x[i]);
            y[i + stride]    = gelu_function<scalar_t>(x[i + stride]);
            y[i + 2*stride]  = gelu_function<scalar_t>(x[i + 2*stride]);
            y[i + 3*stride]  = gelu_function<scalar_t>(x[i + 3*stride]);
        }
        i += 4 * stride;
    }
    // Process any remaining elements
    for (; i < numel; i += stride) {
        y[i] = gelu_function<scalar_t>(x[i]);
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    int threads = 1024;
    int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda_unroll", ([&] {
        gelu_kernel_unroll<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                                            output.data_ptr<scalar_t>(),
                                                            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward with loop unrolling (CUDA)");
}
