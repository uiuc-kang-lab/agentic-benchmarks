#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void optimized_tanh_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    // Calculate global thread ID
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Use grid-stride loop to handle multiple elements per thread
    for (int idx = tid; idx < size; idx += stride) {
        // Compute tanh using fast math intrinsics
        const scalar_t x = input[idx];
        const scalar_t exp2x = __expf(2.0f * x);
        output[idx] = (exp2x - 1.0f) / (exp2x + 1.0f);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int max_blocks = 65535;
    const int blocks = min((input.numel() + threads - 1) / threads, max_blocks);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_tanh_kernel", ([&] {
        optimized_tanh_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Tanh forward (CUDA)");
}