#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>

// Device function to compute softplus in a numerically stable way
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(scalar_t x) {
    return (x > static_cast<scalar_t>(0)) ? (x + log1p(exp(-x))) : log1p(exp(x));
}

// Vectorized kernel processing two elements per thread using reinterpret_cast
// This kernel assumes the input pointer is properly aligned for vectorized loads
// and processes size/2 vectorized packs.
template <typename scalar_t>
__global__ void softplus_kernel_vectorized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int vec_count) {

    // Define a vector type: float2 for float, double2 for double
    using Vec2 = typename std::conditional<std::is_same<scalar_t, float>::value, float2, double2>::type;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Reinterpret the pointers as vectors
    const Vec2* input_vec = reinterpret_cast<const Vec2*>(input);
    Vec2* output_vec = reinterpret_cast<Vec2*>(output);

    for (; idx < vec_count; idx += stride) {
        Vec2 in_val = input_vec[idx];
        Vec2 out_val;

        // Process first element of the vector
        scalar_t x0 = in_val.x;
        scalar_t y0 = compute_softplus(x0);

        // Process second element of the vector
        scalar_t x1 = in_val.y;
        scalar_t y1 = compute_softplus(x1);

        out_val.x = y0;
        out_val.y = y1;
        output_vec[idx] = out_val;
    }
}

// Scalar kernel to process any remaining elements (when size is odd)
template <typename scalar_t>
__global__ void softplus_kernel_scalar(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int start,
    const int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start + idx; i < size; i += stride) {
        scalar_t x = input[i];
        output[i] = compute_softplus(x);
    }
}

// CUDA forward function
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();

    // Use vectorized operations for pairs of elements
    const int vec_pack = 2; // processing two elements at a time
    int vec_count = size / vec_pack;  // number of complete vector packs
    int remainder = size - (vec_count * vec_pack);

    const int threads = 256;
    // Launch configuration for vectorized kernel
    int blocks = (vec_count + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        if (vec_count > 0) {
            softplus_kernel_vectorized<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                vec_count);
        }
        // Process leftover element if any
        if (remainder > 0) {
            int start = vec_count * vec_pack;
            int blocks_rem = (remainder + threads - 1) / threads;
            softplus_kernel_scalar<scalar_t><<<blocks_rem, threads>>>(
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
