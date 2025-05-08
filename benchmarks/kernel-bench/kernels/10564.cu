#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel using grid-stride loops to handle workloads larger than available threads
// Each thread processes one cumulative product chain along the specified dimension

template <typename scalar_t>
__global__ void cumprod_stride_loop_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_batches) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process multiple cumulative product chains using grid-stride loop
    while (idx < total_batches) {
        // Decompose linear index into batch index and inner dimension index
        int batch_idx = idx / stride;
        int in_idx = idx % stride;
        int64_t base_idx = batch_idx * (dim_size * stride) + in_idx;

        scalar_t product = 1;
        // Compute the cumulative product along the dimension
        for (int i = 0; i < dim_size; i++) {
            int64_t offset = base_idx + i * stride;
            product *= input[offset];
            output[offset] = product;
        }

        idx += blockDim.x * gridDim.x;
    }
}

// CUDA forward function

torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);

    // Obtain tensor dimensions and strides
    auto sizes = input.sizes();
    auto strides = input.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t total_batches = input.numel() / dim_size;

    // Kernel launch configuration
    const int threads = 256;
    const int blocks = (total_batches + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_stride_loop", ([&] {
        cumprod_stride_loop_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            dim_size,
            stride,
            total_batches
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward, "Cumulative product with grid stride loops (CUDA)");
}
