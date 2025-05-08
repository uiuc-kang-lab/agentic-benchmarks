#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Modular device function for GELU computation
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_gelu(scalar_t x);

// Specialization for float
template <>
__device__ __forceinline__ float compute_gelu<float>(float x) {
    // GELU(x) = x * 0.5f * (1 + erf(x / sqrt(2)))
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Specialization for double
template <>
__device__ __forceinline__ double compute_gelu<double>(double x) {
    return x * 0.5 * (1.0 + erf(x / 1.4142135623730951));
}

// Struct for 128-bit aligned vectorized loads/stores
// For float: vec_size = 4 (4*4 bytes = 16 bytes); for double: vec_size = 2 (2*8 bytes = 16 bytes)

template <typename scalar_t, int vec_size>
struct AlignedVec {
    scalar_t vals[vec_size];
};

// Modular device function to process a vector chunk
// Applies GELU to each element of the vector load

template <typename scalar_t, int vec_size>
__device__ __forceinline__ void process_vector_chunk(const scalar_t* __restrict__ x,
                                                         scalar_t* __restrict__ y,
                                                         size_t idx) {
    const AlignedVec<scalar_t, vec_size>* x_vec = reinterpret_cast<const AlignedVec<scalar_t, vec_size>*>(x);
    AlignedVec<scalar_t, vec_size>* y_vec = reinterpret_cast<AlignedVec<scalar_t, vec_size>*>(y);
    AlignedVec<scalar_t, vec_size> data = x_vec[idx];

    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
        data.vals[i] = compute_gelu<scalar_t>(data.vals[i]);
    }
    y_vec[idx] = data;
}

// Vectorized kernel: Each thread processes a chunk of vec_size elements

template <typename scalar_t, int vec_size>
__global__ void gelu_kernel_vectorized(const scalar_t* __restrict__ x,
                                        scalar_t* __restrict__ y,
                                        size_t vec_count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    while (index < vec_count) {
        process_vector_chunk<scalar_t, vec_size>(x, y, index);
        index += stride;
    }
}

// Scalar kernel for processing remaining elements

template <typename scalar_t>
__global__ void gelu_kernel_scalar(const scalar_t* __restrict__ x,
                                     scalar_t* __restrict__ y,
                                     size_t start,
                                     size_t numel) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t global_index = start + index;
    if (global_index < numel) {
        y[global_index] = compute_gelu<scalar_t>(x[global_index]);
    }
}

// Forward function callable from Python

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();
    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_modular_cuda", ([&] {
        // Determine vector size: 4 for float (16 bytes) and 2 for double (16 bytes)
        constexpr int V = (sizeof(scalar_t) == 4) ? 4 : 2;
        size_t vec_count = numel / V;

        if (vec_count > 0) {
            int blocks = (vec_count + threads - 1) / threads;
            gelu_kernel_vectorized<scalar_t, V><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                vec_count
            );
        }

        // Process any remaining elements that don't fit into a full vector load
        size_t scalar_start = vec_count * V;
        size_t remainder = numel - scalar_start;
        if (remainder > 0) {
            int blocks = (remainder + threads - 1) / threads;
            gelu_kernel_scalar<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                scalar_start,
                numel
            );
        }
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular GELU activation forward (CUDA)");
}
