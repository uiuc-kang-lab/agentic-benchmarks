#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_hardsigmoid(scalar_t x) {
    scalar_t y = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
    return y < static_cast<scalar_t>(0) ? static_cast<scalar_t>(0) :
           (y > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : y);
}

template <typename scalar_t>
__global__ void optimized_hardsigmoid_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const size_t numel) {
    
    // Use grid-stride loop for better load balancing
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    
    // Process elements with grid-stride loop
    #pragma unroll 4
    for (size_t idx = tid; idx < numel; idx += stride) {
        output[idx] = compute_hardsigmoid(input[idx]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Optimize thread/block configuration for H100
    const int threads_per_block = 256;  // Optimal for H100
    const int min_blocks_per_sm = 4;
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Calculate optimal number of blocks
    const int max_blocks = prop.multiProcessorCount * min_blocks_per_sm;
    const int num_blocks = std::min(
        max_blocks,
        static_cast<int>((numel + threads_per_block - 1) / threads_per_block)
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_hardsigmoid_cuda", ([&] {
        optimized_hardsigmoid_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA)");
}