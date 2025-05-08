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

// CUDA kernel applying SELU using a grid-stride loop for improved mapping of threads to elements.
// This optimizes thread and block indexing, ensuring high occupancy and efficient execution on
// NVIDIA H100 GPUs with CUDA 12.2.

template <typename scalar_t>
__global__ void selu_kernel(const scalar_t* __restrict__ input,
                            scalar_t* __restrict__ output,
                            size_t numel) {
    // SELU parameters
    const scalar_t alpha  = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);

    // Grid-stride loop: each thread processes multiple elements if available.
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < numel;
         idx += blockDim.x * gridDim.x) {
        scalar_t x = input[idx];
        scalar_t value = (x > static_cast<scalar_t>(0)) ? x : alpha * (my_exp(x) - static_cast<scalar_t>(1));
        output[idx] = lambda * value;
    }
}

// Host function that launches the CUDA SELU kernel.
// Exposed to Python via the pybind11 module as "forward".

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    
    // Optimally setting block and grid dimensions for a grid-stride loop.
    const int threads = 1024;
    int blocks = (numel + threads - 1) / threads;

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
