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

// Optimized CUDA kernel with loop unrolling to reduce loop overhead.
template <typename scalar_t>
__global__ void selu_kernel_unroll(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     size_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes four elements per loop iteration using unrolling
    for (size_t i = idx; i < numel; i += stride * 4) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            size_t index = i + j * stride;
            if (index < numel) {
                scalar_t x = input[index];
                scalar_t result = (x > static_cast<scalar_t>(0))
                                  ? x
                                  : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
                output[index] = static_cast<scalar_t>(1.05070098735548049342) * result;
            }
        }
    }
}

// Host function that launches the optimized CUDA SELU kernel
// The grid configuration accounts for the unrolling factor of 4 per thread.
torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    const int threads = 1024;
    // Adjust block count to account for 4 elements processed per thread per iteration
    const int blocks = (numel + threads * 4 - 1) / (threads * 4);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda_unroll", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        scalar_t *output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_unroll<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "Optimized SELU Activation Forward (CUDA) with Loop Unrolling");
}
