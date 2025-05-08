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

// CUDA kernel with loop unrolling to reduce loop overhead.
// Each thread processes 'unroll_factor' elements in a single loop iteration.

template <typename scalar_t>
__global__ void selu_kernel_unroll(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    const int unroll_factor = 4;
    size_t idx = blockIdx.x * blockDim.x * unroll_factor + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < unroll_factor; ++i) {
        if (idx < numel) {
            scalar_t x = input[idx];
            scalar_t y = (x > static_cast<scalar_t>(0))
                             ? x
                             : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
            output[idx] = static_cast<scalar_t>(1.05070098735548049342) * y;
        }
        idx += blockDim.x;
    }
}

// Host function that launches the unrolled CUDA SELU kernel.
// The kernel launch uses adjusted grid dimensions to account for the unroll factor.

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;
    const int unroll_factor = 4;
    const int blocks = (numel + threads * unroll_factor - 1) / (threads * unroll_factor);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        scalar_t *output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_unroll<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Loop Unrolling (CUDA)");
}
