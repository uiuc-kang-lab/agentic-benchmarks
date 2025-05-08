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

// CUDA kernel that applies the SELU activation to each element with memory coalescing and loop unrolling.
template <typename scalar_t>
__global__ void selu_kernel_coalesced_unroll(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      size_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);

    #pragma unroll 4
    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        // Use fast math intrinsics for exponential when using float
        scalar_t exp_x = my_exp(x);
        // Fuse operations to reduce register pressure
        output[i] = lambda * ((x > static_cast<scalar_t>(0)) ? x : alpha * (exp_x - static_cast<scalar_t>(1)));
    }
}

// Host function that launches the CUDA SELU kernel with memory coalescing and loop unrolling.
torch::Tensor selu_forward_coalesced_unroll(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda_coalesced_unroll", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        scalar_t *output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_coalesced_unroll<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward_coalesced_unroll, "SELU Activation Forward Coalesced and Unroll (CUDA)");
}