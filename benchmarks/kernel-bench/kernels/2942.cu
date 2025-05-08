#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tanh_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop to handle multiple elements per thread
    for (int idx = blockIdx.x * blockDim.x + tid; idx < size; idx += stride) {
        output[idx] = tanhf(input[idx]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    // Optimize block and grid size for H100
    const int threads = 256;
    const int max_blocks = 256;
    const int blocks = min(max_blocks, (input.numel() + threads - 1) / threads);
    
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