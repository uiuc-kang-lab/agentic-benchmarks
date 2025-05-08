#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: define an inline exponential function for float and double.
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

// CUDA kernel that applies the SELU activation in an elementwise manner.
// Atomic operations are intentionally omitted because each thread processes independent data,
// eliminating race conditions and avoiding unnecessary global memory contention.
// If an atomic update were required (e.g., for a reduction), a block-level reduction would be used
// to minimize atomic operations to one per block.

template <typename scalar_t>
__global__ void selu_kernel_no_atomic(const scalar_t* __restrict__ input,
                                        scalar_t* __restrict__ output,
                                        size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = __ldg(&input[i]);
        scalar_t res = (x > static_cast<scalar_t>(0))
                           ? x
                           : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = static_cast<scalar_t>(1.05070098735548049342) * res;
    }
}

// Host function that launches the SELU kernel without any unnecessary atomic operations.
// This ensures that each thread writes to a unique output location, thereby avoiding race conditions.

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_no_atomic_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_no_atomic<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward without unnecessary atomic operations (CUDA)");
}
