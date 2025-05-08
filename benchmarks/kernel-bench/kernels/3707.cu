#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel: optimized HardSigmoid using warp-level operations
template <typename scalar_t>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    const unsigned int tid = threadIdx.x;
    const unsigned int wid = tid >> 5;  // Warp ID
    const unsigned int lane = tid & 31;  // Lane within warp
    const unsigned int warp_size = 32;
    const unsigned int idx_base = (blockIdx.x * blockDim.x + wid * warp_size);
    
    // Constants for HardSigmoid
    const scalar_t three = 3.0;
    const scalar_t sixth = 1.0/6.0;
    
    // Process elements in warp-sized chunks
    if (idx_base + lane < numel) {
        scalar_t x = input[idx_base + lane];
        scalar_t y = (x + three) * sixth;
        
        // Use faster intrinsics for min/max operations
        y = max(scalar_t(0.0), min(scalar_t(1.0), y));
        
        output[idx_base + lane] = y;
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