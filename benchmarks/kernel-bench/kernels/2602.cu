#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel using a grid-stride loop and __ldg for read caching
// Combines a robust iteration pattern with simple element-wise ReLU logic

template <typename scalar_t>
__global__ void grid_stride_relu_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop: each thread handles multiple elements if necessary
    for (; idx < size; idx += stride) {
        // Use __ldg for potentially improved caching on read-only data
        scalar_t x = __ldg(&input[idx]);
        output[idx] = (x > static_cast<scalar_t>(0)) ? x : static_cast<scalar_t>(0);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_stride_relu_kernel", ([&] {
        grid_stride_relu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward using grid-stride loop (CUDA)");
}
