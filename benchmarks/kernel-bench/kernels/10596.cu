#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel uses a grid-stride loop to ensure uniform control flow among threads,
// thereby minimizing warp divergence. The loop condition is unrolled to further reduce branch overhead.

template <typename scalar_t>
__global__ void cumprod_kernel_grid_stride(
    scalar_t* output,
    const scalar_t* input,
    const int64_t total_threads,
    const int64_t dim_size,
    const int64_t stride) {

    // Each thread processes one cumulative product task along the target dimension
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_threads; idx += blockDim.x * gridDim.x) {
        // Compute the batch and column indices for the current thread
        int batch = idx / stride;
        int col = idx % stride;
        // Base index for this cumulative product computation
        int base = batch * dim_size * stride + col;
        scalar_t prod = static_cast<scalar_t>(1);

        // Unroll the loop to reduce branch overhead
        #pragma unroll
        for (int i = 0; i < dim_size; i++) {
            int offset = base + i * stride;
            prod *= input[offset];
            output[offset] = prod;
        }
    }
}


torch::Tensor cumprod_cuda_forward_grid_stride(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);

    auto sizes = input.sizes();
    auto strides = input.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();
    // Total cumulative product tasks (each task corresponds to one reduction along the cumulative dimension)
    int64_t total_threads = numel / dim_size;

    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_grid_stride", ([&] {
        cumprod_kernel_grid_stride<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &cumprod_cuda_forward_grid_stride, "Cumulative product forward grid stride (CUDA)");
}
