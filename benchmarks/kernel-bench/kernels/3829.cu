#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    // Use grid-stride loop to handle multiple elements per thread
    const int stride = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure coalesced memory access by having consecutive threads
    // access consecutive memory locations
    while (idx < size) {
        const scalar_t x = input[idx];
        if (x > 20.0) {
            output[idx] = x;
        } else if (x < -20.0) {
            output[idx] = exp(x);
        } else {
            output[idx] = log1p(exp(x));
        }
        idx += stride;
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Optimize block and grid size for better occupancy
    const int threads = 256;
    const int blocks = min(65535, (size + threads - 1) / threads);
    
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