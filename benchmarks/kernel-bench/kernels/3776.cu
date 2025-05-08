#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(const scalar_t x) {
    const scalar_t threshold = static_cast<scalar_t>(20.0);
    return (x > threshold) ? x : 
           (x < -threshold) ? exp(x) : 
           log1p(exp(x));
}

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    // Use grid-stride loop for better occupancy
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < size;
         idx += blockDim.x * gridDim.x) {
        output[idx] = compute_softplus(input[idx]);
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Optimize thread and block configuration
    const int threads = 256;  // Optimal for most GPU architectures
    const int max_blocks = 65535;
    const int blocks = min((size + threads - 1) / threads, max_blocks);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}