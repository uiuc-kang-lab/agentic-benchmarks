#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to perform branchless clamping using arithmetic operations
__device__ inline float branchless_clamp(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ inline double branchless_clamp(double x) {
    return fmin(fmax(x, 0.0), 1.0);
}

// Optimized CUDA kernel for HardSigmoid activation
// Uses branchless arithmetic to minimize warp divergence
// and ensure uniform control flow across threads

template <typename scalar_t>
__global__ void warp_optimized_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                                  scalar_t* __restrict__ output,
                                                  size_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const scalar_t add_const = static_cast<scalar_t>(3);
    const scalar_t div_const = static_cast<scalar_t>(1) / static_cast<scalar_t>(6);

    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t y = (x + add_const) * div_const;
        output[i] = branchless_clamp(y);
    }
}

// Host function to dispatch the kernel
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "warp_optimized_hardsigmoid_cuda", ([&] {
        warp_optimized_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp Optimized HardSigmoid activation forward (CUDA)");
}