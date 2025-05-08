#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

// Store constant value in constant memory for fast access
__constant__ float sqrt_2_inv = 0.7071067811865475f;

// GELU function specialization for float and double
template <typename scalar_t>
__device__ __forceinline__ scalar_t gelu_function(scalar_t x);

template <>
__device__ __forceinline__ float gelu_function<float>(float x) {
    return x * 0.5f * (1.0f + erff(x * sqrt_2_inv));
}

template <>
__device__ __forceinline__ double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x * sqrt_2_inv));
}

// Bulk kernel processes a contiguous chunk of data in vectorized loads
// Each thread processes VEC_SIZE elements uniformly without internal conditionals
template <typename scalar_t, int VEC_SIZE>
__global__ void gelu_bulk_kernel(const scalar_t* __restrict__ input,
                                 scalar_t* __restrict__ output,
                                 size_t vec_count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vec_count) {
        size_t base = i * VEC_SIZE;
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            output[base + j] = gelu_function<scalar_t>(input[base + j]);
        }
    }
}

// Remainder kernel to process tail elements (if any) with minimal divergence
template <typename scalar_t>
__global__ void gelu_remainder_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      size_t start,
                                      size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = start + i;
    if (idx < n) {
        output[idx] = gelu_function<scalar_t>(input[idx]);
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    auto y = torch::empty_like(x);
    size_t n = x.numel();

    const int threads = 32;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_no_warp_divergence_cuda", ([&] {
        if constexpr (std::is_same<scalar_t, float>::value) {
            constexpr int VEC_SIZE = 4;
            size_t bulk_count = n / VEC_SIZE;           // Number of full vector chunks
            size_t bulk_n = bulk_count * VEC_SIZE;        // Total number of elements in bulk processing
            if (bulk_count > 0) {
                int blocks = (bulk_count + threads - 1) / threads;
                gelu_bulk_kernel<scalar_t, VEC_SIZE><<<blocks, threads>>>(
                    x.data_ptr<scalar_t>(),
                    y.data_ptr<scalar_t>(),
                    bulk_count);
            }
            if (bulk_n < n) {
                size_t remainder = n - bulk_n;
                int blocks = (remainder + threads - 1) / threads;
                gelu_remainder_kernel<scalar_t><<<blocks, threads>>>(
                    x.data_ptr<scalar_t>(),
                    y.data_ptr<scalar_t>(),
                    bulk_n, n);
            }
        } else { // double
            constexpr int VEC_SIZE = 2;
            size_t bulk_count = n / VEC_SIZE;
            size_t bulk_n = bulk_count * VEC_SIZE;
            if (bulk_count > 0) {
                int blocks = (bulk_count + threads - 1) / threads;
                gelu_bulk_kernel<scalar_t, VEC_SIZE><<<blocks, threads>>>(
                    x.data_ptr<scalar_t>(),
                    y.data_ptr<scalar_t>(),
                    bulk_count);
            }
            if (bulk_n < n) {
                size_t remainder = n - bulk_n;
                int blocks = (remainder + threads - 1) / threads;
                gelu_remainder_kernel<scalar_t><<<blocks, threads>>>(
                    x.data_ptr<scalar_t>(),
                    y.data_ptr<scalar_t>(),
                    bulk_n, n);
            }
        }
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU forward with minimized warp divergence (CUDA)");
}
