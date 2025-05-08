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

// CUDA kernel applying SELU activation with manual loop unrolling.
// Each thread processes several elements per iteration to reduce loop overhead.

template <typename scalar_t>
__global__ void selu_kernel_manual_unroll(const scalar_t* __restrict__ input,
                                           scalar_t* __restrict__ output,
                                           size_t numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const int unroll_factor = 4;
    size_t i = tid;

    // Unrolled loop: process unroll_factor elements per iteration
    for (; i + (unroll_factor - 1) * stride < numel; i += stride * unroll_factor) {
        #pragma unroll
        for (int j = 0; j < unroll_factor; j++) {
            size_t index = i + j * stride;
            scalar_t x = input[index];
            scalar_t res = (x > static_cast<scalar_t>(0))
                               ? x
                               : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
            output[index] = static_cast<scalar_t>(1.05070098735548049342) * res;
        }
    }

    // Process any remaining elements that don't fit into a full unrolled iteration
    for (; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t res = (x > static_cast<scalar_t>(0))
                           ? x
                           : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = static_cast<scalar_t>(1.05070098735548049342) * res;
    }
}

// Host function that launches the manually unrolled CUDA SELU kernel
// Grid dimensions are computed to cover all elements considering the unroll factor.

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    const int threads = 1024;
    const int unroll_factor = 4;
    const int blocks = (numel + threads * unroll_factor - 1) / (threads * unroll_factor);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda_manual_unroll", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_manual_unroll<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Manual Loop Unrolling (CUDA)");
}
