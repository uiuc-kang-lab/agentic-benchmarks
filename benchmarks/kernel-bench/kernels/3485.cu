#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device GELU function specializations
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

// Optimized indexing kernel
// Using 2D grid and block configuration to improve memory coalescing and load balancing

template <typename scalar_t, int VEC_SIZE>
__global__ void gelu_optimized_kernel(const scalar_t* __restrict__ input,
                                       scalar_t* __restrict__ output,
                                       size_t n) {
    const int tid = blockIdx.x * blockDim.x * VEC_SIZE + threadIdx.x * VEC_SIZE;
    const int yid = blockIdx.y * blockDim.y + threadIdx.y;

    int global_tid = tid + yid * gridDim.x * blockDim.x * VEC_SIZE;

    if (global_tid < n) {
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            scalar_t val = input[global_tid + i];
            output[global_tid + i] = gelu_function(val);
        }
    } else if (global_tid < n) {
        for (int i = 0; i < VEC_SIZE; ++i) {
            int index = global_tid + i;
            if (index < n) {
                scalar_t val = input[index];
                output[index] = gelu_function(val);
            }
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    auto y = torch::empty_like(x);
    const size_t n = x.numel();
    
    constexpr int VEC_SIZE = 4;
    const int threads_x = 32;
    const int threads_y = 8;
    const dim3 threads(threads_x, threads_y);
    const dim3 blocks((n + threads_x * VEC_SIZE - 1) / (threads_x * VEC_SIZE),
                      (n + threads_y - 1) / threads_y);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_forward", [&] {
        gelu_optimized_kernel<scalar_t, VEC_SIZE>
            <<<blocks, threads>>>(x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU forward (CUDA)");
}
