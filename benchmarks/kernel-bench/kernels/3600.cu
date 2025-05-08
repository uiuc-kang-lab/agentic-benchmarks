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
__global__ void selu_kernel_optimized(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, size_t numel) {
    __shared__ scalar_t shared_input[256];
    __shared__ scalar_t shared_output[256];
                                    scalar_t* __restrict__ output,
                                    size_t numel) {
    const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);
    
    const size_t elements_per_thread = 4;
    size_t start_idx = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;
    size_t total_threads = blockDim.x * gridDim.x;
    size_t stride = total_threads * elements_per_thread;

    for (size_t idx = start_idx; idx < numel; idx += stride) {
        #pragma unroll
        for (size_t i = 0; i < elements_per_thread; ++i) {
            size_t element_idx = idx + i;
            if (element_idx < numel) {
                scalar_t x = input[element_idx];
                scalar_t result = (x > static_cast<scalar_t>(0))
                                   ? x
                                   : alpha * (my_exp(x) - static_cast<scalar_t>(1));
                output[element_idx] = lambda * result;
            }
        }
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    const int threads = 256;
    const int blocks = (numel + (threads * 4) - 1) / (threads * 4);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_optimized", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_optimized<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward (Optimized CUDA)");
}
