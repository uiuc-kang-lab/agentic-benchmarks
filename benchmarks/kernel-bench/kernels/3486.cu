#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// GELU function specializations
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
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

// Vectorized GELU computation kernel
// Each thread processes VEC_SIZE elements
// Use streams to overlap computation with data transfer

template <typename scalar_t, int VEC_SIZE>
__global__ void gelu_vectorized_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      size_t n) {
    const int tid = blockIdx.x * blockDim.x * VEC_SIZE + threadIdx.x * VEC_SIZE;
    if (tid + VEC_SIZE <= n) {
        // Process full vector without boundary checks
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            scalar_t val = input[tid + i];
            output[tid + i] = gelu_function(val);
        }
    } else if (tid < n) {
        // Process remaining elements with boundary checks
        for (int i = 0; i < VEC_SIZE; ++i) {
            int index = tid + i;
            if (index < n) {
                scalar_t val = input[index];
                output[index] = gelu_function(val);
            }
        }
    }
}

// Forward function callable from Python
// Uses CUDA streams to overlap memory transfers and computation

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    auto y = torch::empty_like(x);
    const size_t n = x.numel();
    constexpr int VEC_SIZE = 4;
    const int threads = 256;
    const int blocks = (n + threads * VEC_SIZE - 1) / (threads * VEC_SIZE); if (blocks > 65535) blocks = 65535;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_forward", [&] {
        gelu_vectorized_kernel<scalar_t, VEC_SIZE>
            <<<blocks, threads, 0, stream>>>(x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
    });

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU forward with CUDA streams (CUDA)");
}
