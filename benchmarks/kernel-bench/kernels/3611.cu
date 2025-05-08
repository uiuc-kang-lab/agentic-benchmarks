#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t selu_transform(scalar_t x) {
    constexpr scalar_t alpha = 1.67326324235437728481;
    constexpr scalar_t lambda = 1.05070098735548049342;
    
    const scalar_t positive = x;
    const scalar_t negative = alpha * (exp(x) - scalar_t(1));
    return lambda * (x > scalar_t(0) ? positive : negative);
}

template <typename scalar_t>
__global__ void selu_kernel_modular(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < numel; i += stride) {
        output[i] = selu_transform<scalar_t>(input[i]);
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Optimized launch configuration
    const int threads = 512;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        selu_kernel_modular<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward (Modular CUDA)");
}