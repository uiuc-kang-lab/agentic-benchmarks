#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel assigns one thread per output element, ensuring that threads in the same warp
// read consecutive memory locations when iterating over the reduction dimension.
// Input tensor is assumed to be contiguous in memory (C-contiguous), with shape:
// [outer_size, reduce_size, inner_size] when reducing over dimension 'dim'.

template <typename scalar_t>
__global__ void coalesced_sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size) {

    // Each block is organized such that blockIdx.x indexes the outer dimension,
    // and blockIdx.y together with threadIdx.x covers the inner dimension.
    int outer_idx = blockIdx.x;  // Each block in x-dimension corresponds to one outer index
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;  // Thread's inner position

    if (inner_idx < inner_size) {
        // Compute the base index in the flattened input tensor for this (outer, inner) pair
        int64_t base = outer_idx * reduce_size * inner_size + inner_idx;
        scalar_t sum_val = 0;
        
        // Loop over the reduction dimension. For each fixed i, the accesses are contiguous
        // for threads with consecutive inner_idx, ensuring coalesced global memory accesses.
        for (int i = 0; i < reduce_size; i++) {
            sum_val += input[base + i * inner_size];
        }

        // Write the result to the output tensor at the appropriate location
        output[outer_idx * inner_size + inner_idx] = sum_val;
    }
}

// CUDA wrapper function
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Get sizes and compute outer, reduce and inner dimensions
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Prepare output tensor: set the reduced dimension's size to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Configure kernel launch parameters to ensure memory coalescing:
    // We use a 2D grid where:
    //   - grid.x is the outer dimension (each block in x handles one outer index),
    //   - grid.y covers the inner dimension, with each thread handling one output element.
    const int threads = 256; // blockDim.x (must be a multiple of warp size, i.e., 32)
    int grid_y = (inner_size + threads - 1) / threads;
    dim3 blocks(outer_size, grid_y);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        coalesced_sum_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Coalesced thread sum reduction forward (CUDA)");
}
