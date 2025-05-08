#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined kernel: uses a 2D grid over the outer and inner dimensions.
// It caps the grid's y-dimension to avoid launching an excessive number of blocks,
// while using a strided loop to cover the inner dimension if needed.

template <typename scalar_t>
__global__ void combined_coalesced_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t inner_size,
    const int64_t dim_size
) {
    // Each block in x-dimension handles one outer index
    int outer_idx = blockIdx.x;
    
    // Total stride across the inner dimension from the 2D grid
    int total_stride = blockDim.x * gridDim.y;

    // Each thread starts at a unique inner index and may process multiple indices in strides
    for (int inner_idx = blockIdx.y * blockDim.x + threadIdx.x; inner_idx < inner_size; inner_idx += total_stride) {
        int64_t base_offset = outer_idx * dim_size * inner_size;
        
        // Initialize reduction with the first element along the reduced dimension
        scalar_t max_val = input[base_offset + inner_idx];

        // Loop over the reduction dimension
        #pragma unroll
        for (int i = 1; i < dim_size; i++) {
            scalar_t val = input[base_offset + i * inner_size + inner_idx];
            max_val = max(max_val, val);
        }

        // Write the result. The output tensor is conceptually [outer, inner].
        output[outer_idx * inner_size + inner_idx] = max_val;
    }
}


// Host function: computes outer_size and inner_size based on the input shape, sets up a 2D grid configuration,
// and limits the number of blocks in the inner dimension (grid.y) for efficiency. If inner_size is large,
// each thread will loop over multiple inner elements using a strided pattern.

torch::Tensor combined_coalesced_max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Compute outer_size: product of dimensions before the reduction dimension
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    // Compute inner_size: product of dimensions after the reduction dimension
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    // Size along the reduction dimension
    const int64_t dim_size = input.size(dim);

    // Create output tensor with the reduced dimension removed
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const int threads = 256;
    
    // Determine number of blocks in the y-dimension.
    // Instead of launching a grid covering the full inner dimension, we cap it to a maximum
    // number and have each thread process multiple inner indices via a strided loop.
    int blocks_y = (inner_size + threads - 1) / threads;
    const int max_blocks_y = 64;  // tunable parameter to balance occupancy and launch overhead
    if (blocks_y > max_blocks_y) {
        blocks_y = max_blocks_y;
    }

    // Configure a 2D grid: grid.x for outer indices, grid.y for tiling inner indices
    dim3 grid(outer_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "combined_coalesced_max_reduce_forward", ([&] {
        combined_coalesced_max_reduce_kernel<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            inner_size,
            dim_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &combined_coalesced_max_reduce_cuda_forward, "Combined Coalesced Max Reduction forward (CUDA)");
}
