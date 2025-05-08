#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

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

template <typename scalar_t>
__global__ void selu_kernel(const scalar_t* __restrict__ input,
                          scalar_t* __restrict__ output,
                          size_t numel) {
    const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += blockDim.x * gridDim.x) {
        const scalar_t x = input[idx];
        const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
        const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);
        
        // Branchless SELU using step function multiplication
        const scalar_t is_positive = x > 0;
        const scalar_t exp_term = my_exp(x) - static_cast<scalar_t>(1);
        
        // result = is_positive * x + (!is_positive) * (alpha * (exp(x) - 1))
        const scalar_t result = is_positive * x + 
                               (static_cast<scalar_t>(1) - is_positive) * (alpha * exp_term);
        output[idx] = lambda * result;
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        scalar_t *output_ptr = output.data_ptr<scalar_t>();
        selu_kernel<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward (CUDA)");
}