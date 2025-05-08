#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: define an inline exponential function for float and double
template <typename scalar_t>
__device__ __forceinline__ scalar_t my_exp(scalar_t x);

template <>
__device__ __forceinline__ float my_exp<float>(float x) {
    return expf(x);
}

template <>
__device__ __forceinline__ double my_exp<double>(double x) {
    return exp(x);
}

// Compute SELU activation for negative values
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_negative_selu(scalar_t x) {
    constexpr scalar_t alpha = 1.67326324235437728481;
    return alpha * (my_exp(x) - static_cast<scalar_t>(1));
}

// Compute final SELU scaling
template <typename scalar_t>
__device__ __forceinline__ scalar_t apply_selu_scaling(scalar_t x) {
    constexpr scalar_t lambda = 1.05070098735548049342;
    return lambda * x;
}

// Main SELU computation
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_selu(scalar_t x) {
    return apply_selu_scaling(x > static_cast<scalar_t>(0) ? x : compute_negative_selu(x));
}

// Optimized CUDA kernel with modular device functions
template <typename scalar_t>
__global__ void selu_kernel_modular(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   const size_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        // Use __ldg for read-only cached access
        const scalar_t x = __ldg(&input[idx]);
        output[idx] = compute_selu(x);
    }
}

// Host function that launches the modular CUDA SELU kernel
torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_modular<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "Modular SELU Activation Forward (CUDA)");
}