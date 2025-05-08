#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Modular device function for ReLU operation, inlined for performance
template <typename scalar_t>
__device__ __forceinline__ scalar_t relu_val(scalar_t x) {
    return x > static_cast<scalar_t>(0) ? x : static_cast<scalar_t>(0);
}

// CUDA kernel using grid-stride loop that calls the modular ReLU function
template <typename scalar_t>
__global__ void relu_kernel_modularized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x; // Calculate the total number of threads in the grid
    
    for (; idx < size; idx += stride) {
        output[idx] = relu_val(input[idx]);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_modularized", ([&] {
        relu_kernel_modularized<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modularized Optimized ReLU forward (CUDA)");
}
