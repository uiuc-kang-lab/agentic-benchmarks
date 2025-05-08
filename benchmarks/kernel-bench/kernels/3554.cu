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

// CUDA kernel that applies the SELU activation to 4 elements per thread
template <typename scalar_t>
__global__ void selu_kernel(const scalar_t* __restrict__ input,
                          scalar_t* __restrict__ output,
                          size_t numel) {
    const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);

    // Each thread processes 4 elements
    size_t idx = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    
    // Load 4 elements if available
    scalar_t x[4];
    scalar_t result[4];
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (idx + i < numel) {
            x[i] = input[idx + i];
            result[i] = (x[i] > static_cast<scalar_t>(0)) ?
                        x[i] :
                        alpha * (my_exp(x[i]) - static_cast<scalar_t>(1));
            result[i] = lambda * result[i];
            output[idx + i] = result[i];
        }
    }
}

// Host function that launches the CUDA SELU kernel.
torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Adjust grid dimensions for 4-element vectorization
    const int threads = 256;
    const int elements_per_thread = 4;
    const int blocks = (numel + (threads * elements_per_thread) - 1) / (threads * elements_per_thread);

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