#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for ReLU operation
template <typename scalar_t>
__device__ scalar_t relu_op(scalar_t val) {
    return val > static_cast<scalar_t>(0) ? val : static_cast<scalar_t>(0);
}

// CUDA kernel for ReLU activation using modular device function
template <typename scalar_t>
__global__ void relu_kernel_modular(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread processes multiple elements using the device function
    for (int i = idx; i < size; i += stride) {
        output[i] = relu_op(input[i]);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_modular", ([&] {
        relu_kernel_modular<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with modular device function (CUDA)");
}