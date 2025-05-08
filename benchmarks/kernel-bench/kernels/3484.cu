#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Store frequently accessed constants in constant memory
__constant__ float sqrt_2_inv = 0.7071067811865475f;

// Device GELU function specializations
template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x);

template <>
__device__ inline float gelu_function<float>(float x) {
    return x * 0.5f * (1.0f + erff(x * sqrt_2_inv));
}

template <>
__device__ inline double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x * sqrt_2_inv));
}

// Vectorized kernel with constant memory usage
template <typename scalar_t, int VEC_SIZE>
__global__ void gelu_vectorized_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      size_t n) {
    int base_idx = blockIdx.x * blockDim.x * VEC_SIZE + threadIdx.x * VEC_SIZE;
    extern __shared__ scalar_t tile[];
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        int idx = base_idx + i;
        tile[threadIdx.x * VEC_SIZE + i] = (idx < n) ? input[idx] : scalar_t(0);
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        int idx = base_idx + i;
        if (idx < n) {
            scalar_t val = tile[threadIdx.x * VEC_SIZE + i];
            tile[threadIdx.x * VEC_SIZE + i] = gelu_function(val);
        }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        int idx = base_idx + i;
        if (idx < n) {
            output[idx] = tile[threadIdx.x * VEC_SIZE + i];
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