#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using warp-level primitives for potential reduction optimizations
// Note: The warp-level reduction here is for demonstration. It does not alter the element-wise tanh output,
// but illustrates how to replace shared memory reductions with __shfl_down_sync when needed.

template <typename scalar_t>
__global__ void tanh_kernel_warp(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Compute tanh activation for the element
        scalar_t x = input[idx];
        scalar_t y = tanhf(x);
        output[idx] = y;

        // Example warp-level reduction (could be used for aggregating metrics if needed)
        // Here, we compute the warp-sum of y as a demonstration of __shfl_down_sync usage.
        unsigned int mask = __ballot_sync(0xffffffff, idx < size);
        scalar_t warp_sum = y;

        // Perform warp-level reduction
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(mask, warp_sum, offset);
        }

        // The result in warp_sum is the sum of tanh values in the warp.
        // Optionally, thread 0 of each warp could use this value (e.g., for logging or further computation),
        // but we leave it unused to preserve the expected element-wise output.
    }
}

// Forward function wrapping the kernel launch

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_warp", ([&] {
        tanh_kernel_warp<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with warp-level primitives (CUDA)");
}
