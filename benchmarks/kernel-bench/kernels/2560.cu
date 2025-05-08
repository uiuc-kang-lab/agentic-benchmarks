#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Inline device function for ReLU activation
template <typename scalar_t>
__forceinline__ __device__ scalar_t relu_activation(scalar_t val) {
    return val > static_cast<scalar_t>(0) ? val : static_cast<scalar_t>(0);
}

// Optimized CUDA kernel using a grid-stride loop with loop unrolling
template <typename scalar_t>
__global__ void unrolled_grid_stride_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread processes multiple elements in a grid-stride loop
    // Unroll loop to process 4 elements at a time
    #pragma unroll 4
    for (int i = idx; i < size; i += stride) {
        output[i] = relu_activation(input[i]);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "unrolled_grid_stride_relu_kernel", ([&] {
        unrolled_grid_stride_relu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Unrolled Grid Stride ReLU forward (CUDA)");
}
