#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(const scalar_t x) {
    if (x > static_cast<scalar_t>(20.0)) {
        return x;
    } else if (x < static_cast<scalar_t>(-20.0)) {
        return exp(x);
    }
    return log1p(exp(x));
}

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    // Calculate base index for the entire block
    const int block_start = blockIdx.x * blockDim.x;
    const int thread_idx = threadIdx.x;
    const int global_idx = block_start + thread_idx;
    
    // Single boundary check for the block
    if (block_start < size) {
        // Process elements within valid range
        if (global_idx < size) {
            const scalar_t x = input[global_idx];
            output[global_idx] = compute_softplus(x);
        }
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Optimize thread count to be multiple of warp size (32)
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    // Ensure we don't exceed maximum grid size
    const int max_blocks = 65535;
    const int grid_blocks = std::min(blocks, max_blocks);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<grid_blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}