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

// A helper struct for 128-bit aligned vectorized load/store.
// For float, vec_size = 4 (4*4 bytes = 16 bytes); for double, vec_size = 2 (2*8 bytes = 16 bytes).

template <typename scalar_t, int vec_size>
struct AlignedVec {
    scalar_t vals[vec_size];
};

// Vectorized kernel using __ldg for read-only global memory accesses.
// The kernel processes vec_size elements per thread.

template <typename scalar_t, int vec_size>
__global__ void gelu_kernel_vec(const scalar_t* __restrict__ x,
                                 scalar_t* __restrict__ y,
                                 size_t numel) {
    // Number of vectorized chunks
    const size_t vec_count = numel / vec_size;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_count) {
        // Reinterpret pointers to vector type. Assumes pointers are 128-bit aligned.
        const AlignedVec<scalar_t, vec_size>* x_vec = reinterpret_cast<const AlignedVec<scalar_t, vec_size>*>(x);
        AlignedVec<scalar_t, vec_size>* y_vec = reinterpret_cast<AlignedVec<scalar_t, vec_size>*>(y);
        
        // Use __ldg for read-only load of a 128-bit chunk
        AlignedVec<scalar_t, vec_size> data = __ldg(&x_vec[idx]);
        #pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            data.vals[i] = gelu_function<scalar_t>(data.vals[i]);
        }
        y_vec[idx] = data;
    }
}

// Scalar kernel for processing any remaining elements that don't fill a full vector load.

template <typename scalar_t>
__global__ void gelu_kernel_scalar(const scalar_t* __restrict__ x,
                                    scalar_t* __restrict__ y,
                                    size_t start,
                                    size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t index = start + idx;
    if (index < numel) {
        scalar_t val = __ldg(&x[index]);
        y[index] = gelu_function<scalar_t>(val);
    }
}

// Forward function callable from Python.
// It first runs the vectorized kernel on aligned 128-bit chunks then processes any remaining elements with a scalar kernel.

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();
    
    int threads = 1024;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda_optimized", ([&] {
        // Determine vector size based on scalar type: 4 for float (16 bytes) and 2 for double (16 bytes).
        constexpr int V = (sizeof(scalar_t) == 4) ? 4 : 2;
        size_t vec_count = numel / V;

        // Launch vectorized kernel if there is at least one full vector to process.
        if (vec_count > 0) {
            int blocks = (vec_count + threads - 1) / threads;
            gelu_kernel_vec<scalar_t, V><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                                               output.data_ptr<scalar_t>(),
                                                               numel);
        }
        
        // Process remaining elements that do not fit into a full 128-bit load
        size_t rem_start = vec_count * V;
        size_t remainder = numel - rem_start;
        if (remainder > 0) {
            int blocks = (remainder + threads - 1) / threads;
            gelu_kernel_scalar<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                                              output.data_ptr<scalar_t>(),
                                                              rem_start,
                                                              numel);
        }
    }));
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized GELU activation forward (CUDA)");
}
