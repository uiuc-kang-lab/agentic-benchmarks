#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constants in constant memory for fast broadcast access
__constant__ float UPPER_THRESHOLD = 20.0f;
__constant__ float LOWER_THRESHOLD = -20.0f;

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        const scalar_t x = input[idx];
        if (x > UPPER_THRESHOLD) {
            output[idx] = x;
        } else if (x < LOWER_THRESHOLD) {
            output[idx] = exp(x);
        } else {
            output[idx] = log1p(exp(x));
        }
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 512;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}