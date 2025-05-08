#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// In this kernel each thread processes a distinct element. 
// Therefore, no atomic operations are needed to avoid race conditions. 
// Atomic operations would only be used if multiple threads attempted to write to the same global memory location. 
// Here we demonstrate that by using standard global memory writes, thus minimizing any global atomic usage.

// Device inline clamp function specialized for float
__device__ inline float clamp_val(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

// Device inline clamp function specialized for double
__device__ inline double clamp_val(double x) {
    return fmin(fmax(x, 0.0), 1.0);
}

// CUDA kernel for HardSigmoid activation: y = clamp((x + 3)/6, 0, 1)
// Note: Atomic operations are NOT used here because each thread writes to a unique output location, 
// which avoids race conditions and reduces global memory contention.

template <typename scalar_t>
__global__ void atomic_minimal_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                                    scalar_t* __restrict__ output,
                                                    size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    const scalar_t offset = static_cast<scalar_t>(3);
    const scalar_t factor = static_cast<scalar_t>(1) / static_cast<scalar_t>(6);

    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t y = (x + offset) * factor;
        // Clamp the result to [0, 1] using the inline device function
        y = clamp_val(y);

        // Direct global memory write; no atomic operation required since each output index is unique.
        output[i] = y;
    }
}

// Host function to dispatch the kernel

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "atomic_minimal_hardsigmoid_cuda", ([&] {
        atomic_minimal_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA) with minimal atomic operations usage");
}
