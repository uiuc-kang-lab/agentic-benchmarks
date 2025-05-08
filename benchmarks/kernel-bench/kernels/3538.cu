#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: inline exponential functions for float and double
template <typename scalar_t>
__device__ inline scalar_t my_exp(scalar_t x);

template <>
__device__ inline float my_exp<float>(float x) {
    return expf(x);
}

template <>
__device__ inline double my_exp<double>(double x) {
    return exp(x);
}

// SELU activation function
template <typename scalar_t>
__device__ inline scalar_t selu_activate(scalar_t x) {
    // Constants for SELU activation
    const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t scale = static_cast<scalar_t>(1.05070098735548049342);
    scalar_t res = (x > static_cast<scalar_t>(0))
                        ? x
                        : alpha * (my_exp(x) - static_cast<scalar_t>(1));
    return scale * res;
}

// Define vector types to enable vectorized load/store for coalesced memory access
// For float, we use float4 (4 elements, 16 bytes); for double, we use double2 (2 elements, 16 bytes)

template <typename scalar_t>
struct VecType;

template <>
struct VecType<float> {
    using Type = float4;
    static const int vec_size = 4;
};

template <>
struct VecType<double> {
    using Type = double2;
    static const int vec_size = 2;
};

// CUDA kernel with aligned and coalesced memory accesses using vectorized loads/stores
template <typename scalar_t>
__global__ void selu_kernel_aligned_coalesced(const scalar_t* __restrict__ input,
                                                scalar_t* __restrict__ output,
                                                size_t numel) {
    using VecT = typename VecType<scalar_t>::Type;
    const int vec_size = VecType<scalar_t>::vec_size;
    size_t num_vec = numel / vec_size;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Process main vectorized region with coalesced access
    for (size_t i = tid; i < num_vec; i += stride) {
        VecT in_vec = reinterpret_cast<const VecT*>(input)[i];
        VecT out_vec;
        if constexpr (sizeof(scalar_t) == sizeof(float)) {
            out_vec.x = selu_activate<float>(in_vec.x);
            out_vec.y = selu_activate<float>(in_vec.y);
            out_vec.z = selu_activate<float>(in_vec.z);
            out_vec.w = selu_activate<float>(in_vec.w);
        } else {
            out_vec.x = selu_activate<double>(in_vec.x);
            out_vec.y = selu_activate<double>(in_vec.y);
        }
        reinterpret_cast<VecT*>(output)[i] = out_vec;
    }

    // Process remaining elements that don't fit into a full vector load/store
    size_t start = num_vec * vec_size;
    for (size_t i = start + tid; i < numel; i += stride) {
        output[i] = selu_activate<scalar_t>(input[i]);
    }
}

// Host function that launches the CUDA kernel
torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    // Choose block and grid sizes
    const int threads = 256;
    // Determine vector size based on scalar type (4 for float, 2 for double)
    int vec_size = (input.scalar_type() == torch::kFloat) ? 4 : 2;
    int num_vec = numel / vec_size;
    int blocks = (num_vec + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_aligned_coalesced_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_aligned_coalesced<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Aligned Coalesced Memory Access (CUDA)");
}
