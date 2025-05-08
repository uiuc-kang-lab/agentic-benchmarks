#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel: optimized HardSigmoid using warp-level operations
template <typename scalar_t>
__global__ void __launch_bounds__(256) hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    // Use grid-stride loop for better occupancy
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    
    // Constants for HardSigmoid - declare as constexpr for potential compile-time optimization
    constexpr scalar_t three = 3.0;
    constexpr scalar_t sixth = 1.0/6.0;
    
    // Process elements using grid-stride loop
    for (unsigned int idx = tid; idx < numel; idx += stride) {
        scalar_t x = input[idx];
        // Fuse operations to reduce register pressure
        x = (x + three) * sixth;
        // Use built-in functions for min/max
        x = max(scalar_t(0.0), min(scalar_t(1.0), x));
        output[idx] = x;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Optimize block size to multiple of warp size
    const int threads = 256;  // 8 warps per block
    const int blocks = (numel + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda", ([&] {
        hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA)");
}