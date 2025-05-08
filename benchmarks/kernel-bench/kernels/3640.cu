#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Utility clamp functions for branchless clamping
__device__ inline float clamp_val(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ inline double clamp_val(double x) {
    return fmin(fmax(x, 0.0), 1.0);
}

// Combined CUDA kernel for HardSigmoid activation
// Computes y = clamp((x + 3) / 6, 0, 1) using branchless operations
template <typename scalar_t>
__global__ void combined_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                               scalar_t* __restrict__ output,
                                               size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    // Precompute constants as scalars
    const scalar_t offset = static_cast<scalar_t>(3);
    const scalar_t scale = static_cast<scalar_t>(1) / static_cast<scalar_t>(6);

    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        // Compute scaled value and clamp it in one shot
        scalar_t y = (x + offset) * scale;
        output[i] = clamp_val(y);
    }
}

// Host function to launch the kernel
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();

    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "combined_hardsigmoid_cuda", ([&] {
        combined_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined HardSigmoid activation forward (CUDA) with branchless clamping");
}
