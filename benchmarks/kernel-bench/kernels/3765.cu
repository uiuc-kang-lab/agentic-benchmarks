#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp-optimized kernel using warp-level primitives for small reductions

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(const scalar_t x) {
    if (x > static_cast<scalar_t>(20.0)) {
        return x;
    } else if (x < static_cast<scalar_t>(-20.0)) {
        return exp(x);
    }
    return log1p(exp(x));
}

// Kernel function
template <typename scalar_t>
__global__ void softplus_kernel_warp(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        const scalar_t x = input[idx];
        scalar_t result = compute_softplus(x);

        // Use warp-level primitive to perform a simple reduction
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            result += __shfl_down_sync(0xFFFFFFFF, result, offset);
        }

        // Write the result back to the output
        if (threadIdx.x % warpSize == 0) {
            output[idx] = result;
        }
    }
}

// CUDA forward function
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 512;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_warp<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
