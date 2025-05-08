#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized kernel using a grid-stride loop and loop unrolling
// This kernel computes the cumulative product along the specified dimension in a tensor.

template <typename scalar_t>
__global__ void cumprod_kernel_optimized(
    scalar_t* output,
    const scalar_t* input,
    const int64_t total_threads,
    const int64_t dim_size,
    const int64_t stride) {

    // Use grid-stride loop to cover all tasks without warp divergence
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_threads; idx += blockDim.x * gridDim.x) {
        // Determine the batch and inner index
        int batch = idx / stride;
        int in_idx = idx % stride;

        // Base pointer offset for this cumulative product task
        int base = batch * dim_size * stride + in_idx;
        scalar_t prod = static_cast<scalar_t>(1);

        // Unroll loop over dim_size to reduce loop overhead
        #pragma unroll
        for (int i = 0; i < dim_size; i++) {
            int offset = base + i * stride;
            prod *= input[offset];
            output[offset] = prod;
        }
    }
}

// Host function to launch the optimized CUDA kernel

torch::Tensor cumprod_cuda_forward_optimized(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);

    // Retrieve tensor properties
    auto sizes = input.sizes();
    auto strides = input.strides();

    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();

    // Each cumulative product task processes a complete column along the dim dimension
    int64_t total_threads = numel / dim_size;

    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_forward_optimized", ([&] {
        cumprod_kernel_optimized<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            total_threads,
            dim_size,
            stride
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward_optimized, "Optimized Cumulative product forward (CUDA)");
}
