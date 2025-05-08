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
        // Compute base index and initialize pointers with __restrict__ to help compiler optimizations
        int base = batch * dim_size * stride + col;
        scalar_t prod = static_cast<scalar_t>(1);
        const scalar_t* __restrict__ in_ptr = input + base;
        scalar_t* __restrict__ out_ptr = output + base;

        // Unroll the loop to reduce branch overhead and use pointer arithmetic to avoid repeated offset computations
        #pragma unroll
        for (int i = 0; i < dim_size; i++) {
            prod *= in_ptr[i * stride];
            out_ptr[i * stride] = prod;
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
