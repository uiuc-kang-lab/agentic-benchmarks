#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_kernel_strided(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements with stride
    for (int idx = tid; idx < size; idx += stride) {
        const scalar_t val = input[idx];
        output[idx] = val > 0 ? val : 0;
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = std::min(65535, (int)((input.numel() + threads - 1) / threads));
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_strided", ([&] {
        relu_kernel_strided<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward strided (CUDA)");
}