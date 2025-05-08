#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device GELU function implementation for float and double

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

// CUDA kernel using manual loop unrolling to process 4 elements per iteration

template <typename scalar_t>
__global__ void gelu_kernel_unroll(const scalar_t* __restrict__ x,
                                    scalar_t* __restrict__ y,
                                    size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per loop iteration using unrolling
    while (idx + 3 * stride < numel) {
        #pragma unroll
        {
            y[idx] = gelu_function<scalar_t>(x[idx]);
            y[idx + stride] = gelu_function<scalar_t>(x[idx + stride]);
            y[idx + 2 * stride] = gelu_function<scalar_t>(x[idx + 2 * stride]);
            y[idx + 3 * stride] = gelu_function<scalar_t>(x[idx + 3 * stride]);
        }
        idx += stride * 4;
    }
    
    // Process any remaining elements
    while (idx < numel) {
        y[idx] = gelu_function<scalar_t>(x[idx]);
        idx += stride;
    }
}

// Forward function callable from Python

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();
    
    const int threads = 1024;
    int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_unroll_cuda", ([&] {
        gelu_kernel_unroll<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with loop unrolling");
}
