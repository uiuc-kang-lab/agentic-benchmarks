#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device-specific tanh function: use tanhf for float and tanh for double.
template <typename scalar_t>
__device__ inline scalar_t device_tanh(scalar_t x);

template <>
__device__ inline float device_tanh<float>(float x) {
    return tanhf(x);
}

template <>
__device__ inline double device_tanh<double>(double x) {
    return tanh(x);
}

// Optimized kernel using a grid-stride loop to handle arbitrary tensor sizes
// Note: No shared memory is used, so there is no need for __syncthreads(), which avoids unnecessary synchronizations.
template <typename scalar_t>
__global__ void tanh_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    // Calculate base index for the block
    const int base_idx = blockIdx.x * blockDim.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process elements in a grid-stride loop with explicit coalescing
    for (int idx = base_idx + tid; idx < size; idx += stride) {
        // Threads within a warp access consecutive memory locations
        output[idx] = device_tanh(input[idx]);
    }
}

// Host function that launches the kernel
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_optimized", ([&] {
        tanh_kernel_optimized<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Tanh forward (CUDA)");
}
