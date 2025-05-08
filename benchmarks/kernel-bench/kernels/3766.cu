#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Inlined device function to compute softplus
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(const scalar_t x) {
    if (x > static_cast<scalar_t>(20.0)) {
        return x;
    } else if (x < static_cast<scalar_t>(-20.0)) {
        return exp(x);
    }
    return log1p(exp(x));
}

// CUDA kernel using a block-stride loop to evenly distribute workload
template <typename scalar_t>
__global__ void softplus_kernel_blockstride(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple elements via block-stride loop
    for (; idx < size; idx += stride) {
        // Using __ldg to load read-only data through the cache
        scalar_t x = __ldg(&input[idx]);
        output[idx] = compute_softplus(x);
    }
}

// CUDA forward function
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    // Limit the number of blocks to ensure even work distribution
    blocks = blocks < 1024 ? blocks : 1024;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_blockstride<scalar_t><<<blocks, threadsPerBlock>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
