#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Reduced warp divergence with branchless operations
// Apply the HardSigmoid function using branchless conditional assignments

// CUDA kernel with reduced warp divergence for HardSigmoid
template <typename scalar_t>
__global__ void divergence_reduced_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                                      scalar_t* __restrict__ output,
                                                      size_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const scalar_t add_const = static_cast<scalar_t>(3);
    const scalar_t div_const = static_cast<scalar_t>(1) / static_cast<scalar_t>(6);

    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t y = (x + add_const) * div_const;
        y = fminf(fmaxf(y, static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
        output[i] = y;
    }
}

// Host function to dispatch the kernel
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "divergence_reduced_hardsigmoid_cuda", ([&] {
        divergence_reduced_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA) with reduced divergence");
}