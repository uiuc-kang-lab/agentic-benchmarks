#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Branchless formulation of softplus: f(x) = max(x, 0) + log1p(exp(-|x|))
// This formulation avoids conditional branches and hence reduces warp divergence.

__device__ __forceinline__ float branchless_softplus(float x) {
    return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
}

__device__ __forceinline__ double branchless_softplus(double x) {
    return fmax(x, 0.0) + log1p(exp(-fabs(x)));
}

// CUDA kernel using block-stride loop with branchless softplus
template <typename scalar_t>
__global__ void softplus_kernel_branchless(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < size; idx += stride) {
        scalar_t x = input[idx];
        // Use the branchless softplus formulation for uniform control flow
        output[idx] = branchless_softplus(x);
    }
}

// CUDA forward function
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_branchless<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
