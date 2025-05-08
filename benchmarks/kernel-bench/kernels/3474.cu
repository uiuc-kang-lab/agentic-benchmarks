#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x);

template <>
__device__ inline float gelu_function<float>(float x) {
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865475f));
}

template <>
__device__ inline double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x * 0.7071067811865475));
}

template <typename scalar_t, int VEC_SIZE>
__global__ void gelu_vectorized_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      size_t n) {
    const int tid = blockIdx.x * blockDim.x * VEC_SIZE + threadIdx.x * VEC_SIZE;
    if (tid < n) {
        scalar_t values[VEC_SIZE];
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            if (tid + i < n) values[i] = input[tid + i];
        }
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            if (tid + i < n) values[i] = gelu_function(values[i]);
        }
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            if (tid + i < n) output[tid + i] = values[i];
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    auto y = torch::empty_like(x);
    const size_t n = x.numel();
    
    constexpr int VEC_SIZE = 4;
    const int threads = 256;
    const int blocks = (n + threads * VEC_SIZE - 1) / (threads * VEC_SIZE);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_forward", [&] {
        gelu_vectorized_kernel<scalar_t, VEC_SIZE>
            <<<blocks, threads>>>(x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU forward (CUDA)");
}