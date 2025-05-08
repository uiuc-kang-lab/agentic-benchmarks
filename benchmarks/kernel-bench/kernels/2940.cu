#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tanh_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process multiple elements per thread to reduce overall thread synchronization needs
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        output[i] = tanhf(input[i]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = min(65535, (input.numel() + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "tanh_kernel", ([&] {
        tanh_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward (CUDA)");
}