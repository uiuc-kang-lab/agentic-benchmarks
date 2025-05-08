#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>

// Numerically stable softplus function
template <typename scalar_t>
__device__ __forceinline__ scalar_t softplus_fn(scalar_t x) {
    if (x > static_cast<scalar_t>(20.0)) {
        return x;
    } else if (x < static_cast<scalar_t>(-20.0)) {
        return exp(x);
    } else {
        return log1p(exp(x));
    }
}

// Traits to define vectorized types for coalesced memory access
template <typename T>
struct vectorized_traits;

// For float, process 4 elements at a time using float4 (16 bytes)
template <>
struct vectorized_traits<float> {
    using VecType = float4;
    static constexpr int vec_size = 4;
};

// For double, process 2 elements at a time using double2 (16 bytes)
template <>
struct vectorized_traits<double> {
    using VecType = double2;
    static constexpr int vec_size = 2;
};

// Kernel that processes the bulk of the tensor using vectorized loads/stores
// Ensures memory coalescing by having consecutive threads load contiguous memory
template <typename scalar_t>
__global__ void softplus_kernel_vectorized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int n_vec) {

    using VecType = typename vectorized_traits<scalar_t>::VecType;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Reinterpret pointers as vectorized pointers
    const VecType* __restrict__ input_vec = reinterpret_cast<const VecType*>(input);
    VecType* __restrict__ output_vec = reinterpret_cast<VecType*>(output);

    for (int i = idx; i < n_vec; i += stride) {
        VecType vec = input_vec[i];
        if constexpr (std::is_same<scalar_t, float>::value) {
            vec.x = softplus_fn(vec.x);
            vec.y = softplus_fn(vec.y);
            vec.z = softplus_fn(vec.z);
            vec.w = softplus_fn(vec.w);
        } else { // double
            vec.x = softplus_fn(vec.x);
            vec.y = softplus_fn(vec.y);
        }
        output_vec[i] = vec;
    }
}

// Scalar kernel to process any remaining elements (in case total size is not a multiple of vec_size)
template <typename scalar_t>
__global__ void softplus_kernel_scalar(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int start,
    const int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = start + idx; i < size; i += stride) {
        output[i] = softplus_fn(input[i]);
    }
}

// CUDA forward function
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        // Determine vector size based on the scalar type
        const int vec_size = vectorized_traits<scalar_t>::vec_size;
        int n_vec = size / vec_size;
        int remainder = size - n_vec * vec_size;

        const int threads = 256;
        int blocks = (n_vec + threads - 1) / threads;

        // Launch vectorized kernel if there are full vector packs
        if (n_vec > 0) {
            softplus_kernel_vectorized<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n_vec);
        }
        // Process any remaining elements with a scalar kernel
        if (remainder > 0) {
            int start = n_vec * vec_size;
            int blocks_scalar = (remainder + threads - 1) / threads;
            softplus_kernel_scalar<scalar_t><<<blocks_scalar, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                start,
                size);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
