#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Modular device function for ReLU operation
// Improves readability and reusability by separating logic
// __device__ makes it callable from other device functions/kernels
__device__ float relu_device(float val) {
    return val > 0 ? val : 0;
}

// Optimized CUDA kernel using a grid-stride loop with modular device function
// This kernel calls the device function for the ReLU operation
template <typename scalar_t>
__global__ void modular_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread processes multiple elements in a grid-stride loop
    for (int i = idx; i < size; i += stride) {
        output[i] = relu_device(input[i]);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "modular_relu_kernel", ([&] {
        modular_relu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular ReLU forward (CUDA) using device function");
}
