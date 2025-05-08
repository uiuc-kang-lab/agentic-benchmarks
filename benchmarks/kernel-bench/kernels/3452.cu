#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device GELU function specializations for float and double

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

// Unified CUDA kernel that applies GELU activation with possibility for vectorized access
template <typename scalar_t, typename vec_t>
__global__ void gelu_kernel_vectorized(const scalar_t* __restrict__ x,
                                        scalar_t* __restrict__ y,
                                        size_t num_vec, size_t vec_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec) {
        const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);
        vec_t* y_vec = reinterpret_cast<vec_t*>(y);

        vec_t val = x_vec[idx];
        vec_t res;

        // Apply GELU function element-wise within the vector
        #pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            res[i] = gelu_function<scalar_t>(val[i]);
        }
        y_vec[idx] = res;
    }
}

// Kernel for any remaining elements
__global__ void gelu_kernel_scalar(const float* __restrict__ x, float* __restrict__ y, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        scalar_t val = static_cast<scalar_t>(x[idx]);
        y[idx] = gelu_function<scalar_t>(val);
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    auto output = torch::empty_like(x);
    size_t total = x.numel();
    const int threads = 1024;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_vectorized_gelu", ([&] {
        auto x_ptr = x.data_ptr<scalar_t>();
        auto y_ptr = output.data_ptr<scalar_t>();

        // Determine vectorization factor: 4 for float, 2 for double
        int vec_factor = std::is_same<scalar_t, float>::value ? 4 : 2;

        // Check if pointers are aligned and if vectorized processing is applicable
        if (total >= vec_factor && (reinterpret_cast<uintptr_t>(x_ptr) % (sizeof(scalar_t) * vec_factor) == 0)) {

            size_t num_full_vec = total / vec_factor;
            size_t remainder = total % vec_factor;
            int blocks = (num_full_vec + threads - 1) / threads;

            if (std::is_same<scalar_t, float>::value) {
                gelu_kernel_vectorized<scalar_t, float4><<<blocks, threads>>>(x_ptr, y_ptr, num_full_vec, vec_factor);
            } else { // double
                gelu_kernel_vectorized<scalar_t, double2><<<blocks, threads>>>(x_ptr, y_ptr, num_full_vec, vec_factor);
            }

            if (remainder > 0) {
                int rem_blocks = (remainder + threads - 1) / threads;
                gelu_kernel_scalar<<<rem_blocks, threads>>>(x_ptr + num_full_vec * vec_factor, y_ptr + num_full_vec * vec_factor, remainder);
            }
        } else {
            int blocks = (total + threads - 1) / threads;
            gelu_kernel_scalar<<<blocks, threads>>>(x_ptr, y_ptr, total);
        }
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) optimized with vectorization");
}