#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// Device GELU function specializations for float and double

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

// Fallback scalar kernel for any remaining elements
template <typename scalar_t>
__global__ void gelu_kernel_scalar(const scalar_t* __restrict__ x,
                                    scalar_t* __restrict__ y,
                                    size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        scalar_t val = x[idx];
        y[idx] = gelu_function<scalar_t>(val);
    }
}

// Vectorized kernel for floats using float4 to load/store 4 elements at once
__global__ void gelu_kernel_vectorized_float(const float* __restrict__ x,
                                               float* __restrict__ y,
                                               size_t num_vec) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec) {
        // Reinterpret pointers as float4 for vectorized access
        const float4* x_vec = reinterpret_cast<const float4*>(x);
        float4* y_vec = reinterpret_cast<float4*>(y);

        float4 val = x_vec[idx];
        float4 res;
        float4 res;
        res.x = gelu_function<float>(val.x);
        res.y = gelu_function<float>(val.y);
        res.z = gelu_function<float>(val.z);
        res.w = gelu_function<float>(val.w);
        y_vec[idx] = res;
    }
}

// Vectorized kernel for doubles using double2 to load/store 2 elements at once
__global__ void gelu_kernel_vectorized_double(const double* __restrict__ x,
                                                double* __restrict__ y,
                                                size_t num_vec) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec) {
        const double2* x_vec = reinterpret_cast<const double2*>(x);
        double2* y_vec = reinterpret_cast<double2*>(y);

        double2 val = x_vec[idx];
        double2 res;
        res.x = gelu_function<double>(val.x);
        res.y = gelu_function<double>(val.y);
        y_vec[idx] = res;
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    auto output = torch::empty_like(x);
    size_t total = x.numel();
    const int threads = 1024;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "aligned_gelu_cuda", ([&] {
        auto x_ptr = x.data_ptr<scalar_t>();
        auto y_ptr = output.data_ptr<scalar_t>();

        // Determine vectorization factor: 4 for float, 2 for double
        int vec_factor = std::is_same<scalar_t, float>::value ? 4 : 2;

        // Check if the pointer is properly aligned for vectorized loads/stores
        if (total >= (size_t)vec_factor &&
            (reinterpret_cast<uintptr_t>(x_ptr) % (sizeof(scalar_t) * vec_factor) == 0)) {

            size_t num_full_vec = total / vec_factor;
            size_t remainder = total % vec_factor;

            int blocks = (num_full_vec + threads - 1) / threads;
            if (num_full_vec > 0) {
                if (std::is_same<scalar_t, float>::value) {
                    gelu_kernel_vectorized_float<<<blocks, threads>>>(
                        reinterpret_cast<const float*>(x_ptr),
                        reinterpret_cast<float*>(y_ptr),
                        num_full_vec);
                } else { // double
                    gelu_kernel_vectorized_double<<<blocks, threads>>>(
                        reinterpret_cast<const double*>(x_ptr),
                        reinterpret_cast<double*>(y_ptr),
                        num_full_vec);
                }
            }

            // Process any remaining elements with the scalar kernel
            if (remainder > 0) {
                int rem_blocks = (remainder + threads - 1) / threads;
                gelu_kernel_scalar<scalar_t><<<rem_blocks, threads>>>(x_ptr + num_full_vec * vec_factor,
                                                                        y_ptr + num_full_vec * vec_factor,
                                                                        remainder);
            }
        } else {
            // If not aligned or not enough elements, fallback to scalar kernel
            int blocks = (total + threads - 1) / threads;
            gelu_kernel_scalar<scalar_t><<<blocks, threads>>>(x_ptr, y_ptr, total);
        }
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with aligned memory access");
}
