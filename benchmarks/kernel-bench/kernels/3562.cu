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

// CUDA kernel implementing the SELU activation with optimized thread and block indexing.
template <typename scalar_t>
__global__ void selu_kernel(const scalar_t* __restrict__ input,
                            scalar_t* __restrict__ output,
                            size_t numel) {
    // Calculate global thread index
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // SELU parameters
    const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);

    // Grid-stride loop for efficient memory access
    for (; idx < numel; idx += stride) {
        scalar_t x = input[idx];
        scalar_t result = (x > static_cast<scalar_t>(0)) 
                          ? x 
                          : alpha * (my_exp(x) - static_cast<scalar_t>(1));
        output[idx] = lambda * result;
    }
}

// Host function that launches the CUDA SELU kernel.
torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 512;  // Optimal thread count per block
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