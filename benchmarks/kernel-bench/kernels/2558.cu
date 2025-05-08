#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Inline device function for ReLU activation
template <typename scalar_t>
__forceinline__ __device__ scalar_t relu_op(scalar_t x) {
    return x > static_cast<scalar_t>(0) ? x : static_cast<scalar_t>(0);
}

// CUDA kernel with manual loop unrolling using #pragma unroll
template <typename scalar_t>
__global__ void unroll_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int i = idx;

    // Process four elements per iteration when possible using loop unrolling
    while (i + 3 * stride < size) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int pos = i + j * stride;
            output[pos] = relu_op(input[pos]);
        }
        i += 4 * stride;
    }

    // Process any remaining elements
    for (; i < size; i += stride) {
        output[i] = relu_op(input[i]);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t total_elements = input.numel();
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "unroll_relu_kernel", ([&] {
        unroll_relu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            total_elements
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Unrolled ReLU forward (CUDA)");
}
