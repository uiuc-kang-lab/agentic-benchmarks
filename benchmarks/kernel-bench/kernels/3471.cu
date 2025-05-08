#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

// Device GELU function specializations
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

// Fallback elementwise kernel that applies GELU activation per element.
template <typename scalar_t>
__global__ void gelu_kernel(const scalar_t* __restrict__ x,
                            scalar_t* __restrict__ y,
                            size_t numel) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        scalar_t val = x[index];
        y[index] = gelu_function<scalar_t>(val);
    }
}

// Vectorized GELU kernel for float using float4 to process 4 elements concurrently.
__global__ void gelu_kernel_vectorized_float(const float* __restrict__ x,
                                             float* __restrict__ y,
                                             size_t num_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec) {
        // Load 4 consecutive floats
        float4 in = reinterpret_cast<const float4*>(x)[idx];
        in.x = in.x * 0.5f * (1.0f + erff(in.x / 1.4142135623730951f));
        in.y = in.y * 0.5f * (1.0f + erff(in.y / 1.4142135623730951f));
        in.z = in.z * 0.5f * (1.0f + erff(in.z / 1.4142135623730951f));
        in.w = in.w * 0.5f * (1.0f + erff(in.w / 1.4142135623730951f));
        // Write back 4 consecutively processed elements
        reinterpret_cast<float4*>(y)[idx] = in;
    }
}

// Vectorized GELU kernel for double using double2 to process 2 elements concurrently.
__global__ void gelu_kernel_vectorized_double(const double* __restrict__ x,
                                              double* __restrict__ y,
                                              size_t num_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec) {
        // Load 2 consecutive doubles
        double2 in = reinterpret_cast<const double2*>(x)[idx];
        in.x = in.x * 0.5 * (1.0 + erf(in.x / 1.4142135623730951));
        in.y = in.y * 0.5 * (1.0 + erf(in.y / 1.4142135623730951));
        // Write back 2 consecutively processed elements
        reinterpret_cast<double2*>(y)[idx] = in;
    }
}

// Forward function callable from Python. It uses vectorized loads/stores for bulk elements
// to ensure memory coalescing and processes any leftover tail elements with a fallback kernel.
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(x);
    size_t numel = x.numel();
    if (numel == 0) {
        return output;
    }

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda", ([&] {
        // Set the vectorization factor: 4 for float, 2 for double
        const int factor = (sizeof(scalar_t) == 4) ? 4 : 2;
        size_t num_vec = numel / factor;
        size_t tail = numel % factor;

        // Launch vectorized kernel for the main bulk if possible
        if (num_vec > 0) {
            int threads = 1024;
            int blocks = (num_vec + threads - 1) / threads;
            if (sizeof(scalar_t) == 4) {
                gelu_kernel_vectorized_float<<<blocks, threads>>>(
                    x.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    num_vec);
            } else {
                gelu_kernel_vectorized_double<<<blocks, threads>>>(
                    x.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    num_vec);
            }
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
        }

        // Process any tail elements that weren't a full vector load
        if (tail > 0) {
            int threads_tail = 1024;
            int blocks_tail = (tail + threads_tail - 1) / threads_tail;
            gelu_kernel<scalar_t><<<blocks_tail, threads_tail>>>(
                x.data_ptr<scalar_t>() + num_vec * factor,
                output.data_ptr<scalar_t>() + num_vec * factor,
                tail);
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized GELU activation forward (CUDA) with coalesced memory accesses");
}
