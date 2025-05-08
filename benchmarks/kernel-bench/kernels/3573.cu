#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Declare constant memory for SELU parameters
__constant__ float d_alpha_f = 1.67326324235437728481f;
__constant__ float d_lambda_f = 1.05070098735548049342f;
__constant__ double d_alpha_d = 1.67326324235437728481;
__constant__ double d_lambda_d = 1.05070098735548049342;

// Device helper: define an inline exponential function for float and double.
template <typename scalar_t>
__device__ inline scalar_t my_exp(scalar_t x);

template <>
__device__ inline float my_exp<float>(float x) {
    return __expf(x);
}

template <>
__device__ inline double my_exp<double>(double x) {
    return exp(x);
}

// Template functions to get SELU parameters from constant memory
template <typename scalar_t>
__device__ inline scalar_t get_alpha();

template <>
__device__ inline float get_alpha<float>() {
    return d_alpha_f;
}

template <>
__device__ inline double get_alpha<double>() {
    return d_alpha_d;
}

template <typename scalar_t>
__device__ inline scalar_t get_lambda();

template <>
__device__ inline float get_lambda<float>() {
    return d_lambda_f;
}

template <>
__device__ inline double get_lambda<double>() {
    return d_lambda_d;
}

// Optimized CUDA kernel that applies the SELU activation to each element.
template <typename scalar_t>
__global__ void selu_kernel(const scalar_t* __restrict__ input,
                            scalar_t* __restrict__ output,
                            size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    
    // Get constants from constant memory
    const scalar_t alpha = get_alpha<scalar_t>();
    const scalar_t lambda = get_lambda<scalar_t>();
    
    // Grid-stride loop
    for (size_t i = idx; i < numel; i += stride) {
        const scalar_t x = input[i];
        scalar_t result = (x > static_cast<scalar_t>(0))
                              ? lambda * x
                              : lambda * alpha * (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = result;
    }
}

// Host function that launches the optimized CUDA SELU kernel.
torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Optimize thread and block count for H100
    const int threads = 256;
    const int blocks = std::min(65535, (int)((numel + threads - 1) / threads));

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