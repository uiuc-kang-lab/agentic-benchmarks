#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(scalar_t x) {
    // Early exit for positive large values
    if (x > 20.0) {
        return x;
    }
    
    // Cache exp(x) computation since it's used in both remaining branches
    const scalar_t exp_x = exp(x);
    
    // For very negative values, exp(x) is the same as log1p(exp(x))
    if (x < -20.0) {
        return exp_x;
    }
    
    // Default case using cached exp_x
    return log1p(exp_x);
}

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    // Calculate initial position and stride
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements in a grid-stride pattern
    for (int idx = tid; idx < size; idx += stride) {
        const scalar_t x = input[idx];
        output[idx] = compute_softplus(x);
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Optimize thread and block count for H100
    const int threads = 256;  // Good warp alignment
    // Use multiple of SM count (132 on H100) for better distribution
    // but cap at a reasonable maximum
    const int sm_count = 132;
    const int blocks_per_sm = 2;
    const int blocks = min(sm_count * blocks_per_sm, 
                         (size + threads - 1) / threads);

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