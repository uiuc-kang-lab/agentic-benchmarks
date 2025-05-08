#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: define an inline exponential function for float and double.
template <typename scalar_t>
__device__ inline scalar_t my_exp(scalar_t x);

template <>
__device__ inline float my_exp<float>(float x) {
    return __expf(x);  // Using fast math intrinsic
}

template <>
__device__ inline double my_exp<double>(double x) {
    return exp(x);  // Keep regular exp for double precision
}

// CUDA kernel applying SELU activation with manual loop unrolling
template <typename scalar_t>
__global__ void selu_kernel_unrolled(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      size_t numel) {
    const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);
    
    // Each thread processes multiple elements using grid-stride loop
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Manually unroll the loop by a factor of 4
    for (; i + 3 * stride < numel; i += 4 * stride) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            size_t index = i + j * stride;
            scalar_t x = input[index];
            scalar_t result = (x > static_cast<scalar_t>(0))
                                ? x
                                : alpha * (my_exp(x) - static_cast<scalar_t>(1));
            output[index] = lambda * result;
        }
    }
    // Process any remaining elements
    for (; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t result = (x > static_cast<scalar_t>(0))
                            ? x
                            : alpha * (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = lambda * result;
    }
}

// Host function launching the unrolled CUDA SELU kernel
torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_unrolled", ([&] {
        selu_kernel_unrolled<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward Unrolled (CUDA)");
}
