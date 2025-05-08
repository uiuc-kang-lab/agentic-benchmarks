#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel optimized to minimize warp divergence

__global__ void tanh_kernel_warp_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int size) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        // Using ternary operator to ensure uniform control flow
        float val = input[i];
        output[i] = (val < -3.0f) ? -1.0f : ((val > 3.0f) ? 1.0f : tanhf(val));
    }
}

// Forward function wrapping the kernel launch

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_warp_optimized", ([&] {
        tanh_kernel_warp_optimized<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with warp divergence optimization (CUDA)");
}