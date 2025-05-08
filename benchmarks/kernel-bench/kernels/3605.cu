#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Store constants in constant memory for faster access
__constant__ float d_alpha = 1.67326324235437728481f;
__constant__ float d_lambda = 1.05070098735548049342f;
__constant__ double d_alpha_double = 1.67326324235437728481;
__constant__ double d_lambda_double = 1.05070098735548049342;

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

// CUDA kernel that applies the SELU activation to each element with constant memory access.
template <typename scalar_t>
__global__ void selu_kernel_const(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const scalar_t alpha = sizeof(scalar_t) == sizeof(float) ? d_alpha : d_alpha_double;
    const scalar_t lambda = sizeof(scalar_t) == sizeof(float) ? d_lambda : d_lambda_double;

    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t result = (x > static_cast<scalar_t>(0))
                              ? x
                              : alpha * (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = lambda * result;
    }
}

// Host function that launches the CUDA SELU kernel with constant memory access.
torch::Tensor selu_forward_const(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 512;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda_const", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        scalar_t *output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_const<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward_const, "SELU Activation Forward Constant Memory (CUDA)");
}