#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tanh_kernel_unrolled(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per iteration
    const int unroll_factor = 4;
    const int unrolled_size = size - (size % unroll_factor);
    
    // Main loop with manual unrolling
    for (int idx = tid * unroll_factor; idx < unrolled_size; idx += stride * unroll_factor) {
        output[idx] = tanhf(input[idx]);
        output[idx + 1] = tanhf(input[idx + 1]);
        output[idx + 2] = tanhf(input[idx + 2]);
        output[idx + 3] = tanhf(input[idx + 3]);
    }
    
    // Handle remaining elements
    for (int idx = unrolled_size + tid; idx < size; idx += stride) {
        output[idx] = tanhf(input[idx]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = std::min(256, (int)((input.numel() + threads - 1) / threads));
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "tanh_kernel_unrolled", ([&] {
        tanh_kernel_unrolled<scalar_t><<<blocks, threads>>>(
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