#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel performs max reduction over one dimension with configurable block size
// allowing experimentation with different block sizes (e.g. 32, 64, 128, 256, 512) for optimal performance

template <typename scalar_t>
__global__ void blocksize_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Each thread computes one output element corresponding to an inner dimension index
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (inner_idx >= inner_size) return;
    int outer_idx = blockIdx.x;
    const int64_t input_offset = outer_idx * dim_size * inner_size + inner_idx;

    scalar_t max_val = input[input_offset];
    for (int i = 1; i < dim_size; i++) {
        scalar_t val = input[input_offset + i * inner_size];
        max_val = max(max_val, val);
    }
    output[outer_idx * inner_size + inner_idx] = max_val;
}

// The forward function now accepts a block_size parameter (default: 256) to allow experimentation
// with different configurations to optimize occupancy and throughput on the H100 GPU.

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim, int block_size = 256) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Compute outer size (product of dimensions before the reduction dimension)
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    // Compute inner size (product of dimensions after the reduction dimension)
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    const int64_t dim_size = input.size(dim);

    // Prepare output tensor by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Set the block and grid dimensions using the configurable block size
    int threads = block_size;
    int blocks_y = (inner_size + threads - 1) / threads;
    dim3 grid(outer_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        blocksize_max_reduce_kernel<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward,
          "Max reduction forward (CUDA) with configurable block size",
          py::arg("input"), py::arg("dim"), py::arg("block_size") = 256);
}
