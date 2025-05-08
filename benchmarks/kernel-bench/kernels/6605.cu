#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel assumes the input tensor is conceptually [outer, dim, inner] where:
//  outer_size = product of dimensions before the reduced dimension
//  dim_size   = size of the reduced dimension
//  inner_size = product of dimensions after the reduced dimension
// Each block in the x-dimension handles one outer index, and blocks in the y-dimension tile the inner dimension.

template <typename scalar_t>
__global__ void unroll_coalesced_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t inner_size,
    const int64_t dim_size
) {
    // Determine which outer index this block is working on
    int outer_idx = blockIdx.x;
    
    // Determine the tile index in the inner dimension
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (inner_idx >= inner_size) return;

    // Base offset for this outer index
    int64_t base_offset = outer_idx * dim_size * inner_size;

    // Initialize maximum value with the first element in the reduction dimension
    scalar_t max_val = input[base_offset + inner_idx];

    // Loop over the reduction dimension; note that for each i, the memory access
    // is to a contiguous block of memory for threads in the same warp, ensuring coalescing.
    #pragma unroll 4
    for (int i = 1; i < dim_size; i++) {
        scalar_t val = input[base_offset + i * inner_size + inner_idx];
        max_val = max(max_val, val);
    }

    // Write the result to output. The output tensor is conceptually [outer, inner].
    output[outer_idx * inner_size + inner_idx] = max_val;
}

// This function computes the outer_size and inner_size from the input shape, similar to the reference implementation,
// but then it launches a 2D grid that ensures coalesced memory accesses along the inner dimension.

torch::Tensor unroll_coalesced_max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Handle negative dimension
    if (dim < 0) dim += input.dim();

    // Calculate outer_size: product of sizes before the 'dim' dimension
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }
    
    // Calculate inner_size: product of sizes after the 'dim' dimension
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }
    
    // Size along the reduction dimension
    const int64_t dim_size = input.size(dim);

    // Create the output tensor by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Configure block and grid sizes.
    // Use a 2D grid: grid.x = outer_size; grid.y covers the inner dimension tiled by the block.
    const int threads = 256;
    int blocks_y = (inner_size + threads - 1) / threads;
    dim3 grid(outer_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "unroll_coalesced_max_reduce_forward", ([&] {
        unroll_coalesced_max_reduce_kernel<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            inner_size,
            dim_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &unroll_coalesced_max_reduce_cuda_forward, "Unroll and coalesced Max reduction forward (CUDA)");
}