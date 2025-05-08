#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ReLU activation using stride loops
template <typename scalar_t>
__global__ void relu_kernel_stride_optimized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread using stride
    for (int idx = tid; idx < size; idx += stride) {
        const scalar_t val = input[idx];
        output[idx] = val > 0 ? val : 0;
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = std::min(65535, int((input.numel() + threads - 1) / threads));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_stride_optimized", ([&] {
        relu_kernel_stride_optimized<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with stride optimization (CUDA)");
}