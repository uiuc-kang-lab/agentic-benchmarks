#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Defined optimal block size for H100 GPU
const int OPTIMAL_THREADS = 512;

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        const scalar_t x = input[idx];
        if (x > 20.0) {
            output[idx] = x;
        } else if (x < -20.0) {
            output[idx] = exp(x);
        } else {
            output[idx] = log1p(exp(x));
        }
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int blocks = (size + OPTIMAL_THREADS - 1) / OPTIMAL_THREADS;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<blocks, OPTIMAL_THREADS>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}