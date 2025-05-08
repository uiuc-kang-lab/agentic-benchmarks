#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ReLU activation with optimized synchronization
template <typename scalar_t>
__global__ void relu_kernel_optimized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
        const scalar_t val = input[idx];
        output[idx] = val > 0 ? val : 0;
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_optimized", ([&] {
        relu_kernel_optimized<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward optimized (CUDA)");
}