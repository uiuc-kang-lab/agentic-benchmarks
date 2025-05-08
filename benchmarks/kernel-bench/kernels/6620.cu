#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Structure to hold the frequently accessed parameters in constant memory
struct ReduceParams {
    int64_t dim_size;
    int64_t inner_size;
};

// Declare constant memory for the reduce parameters
__constant__ ReduceParams c_params;

// Combined kernel that uses constant memory for parameters and coalesced memory accesses
// The input tensor is conceptually [outer, dim, inner] and the reduction is performed over the 'dim' dimension.
// Each block in grid.x processes one 'outer' index and the blocks in grid.y tile the inner dimension.
// We use a 2D grid and each thread handles one inner index, looping over the reduction dimension.

template <typename scalar_t>
__global__ void combined_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size
) {
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

    // Ensure we are within the valid inner dimension
    if (inner_idx >= c_params.inner_size) return;

    // Compute the base offset for this outer index
    int64_t base_offset = outer_idx * c_params.dim_size * c_params.inner_size;

    // Initialize the maximum value with the first element in the reduction dimension
    scalar_t max_val = input[base_offset + inner_idx];

    // Loop over the reduction dimension (starting from 1 since we've already loaded the first element)
    for (int i = 1; i < c_params.dim_size; i++) {
        scalar_t val = input[base_offset + i * c_params.inner_size + inner_idx];
        max_val = max(max_val, val);
    }

    // Write the result. The output tensor is conceptually [outer, inner].
    output[outer_idx * c_params.inner_size + inner_idx] = max_val;
}

// Host function to set up parameters, copy them to constant memory, and launch the kernel
// It computes outer_size and inner_size from the input shape and removes the reduced dimension

torch::Tensor const_aligned_max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Handle negative dimension
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

    // The size along the reduction dimension
    const int64_t dim_size = input.size(dim);

    // Copy reduction parameters into constant memory for fast access during kernel execution
    ReduceParams params = {dim_size, inner_size};
    cudaMemcpyToSymbol(c_params, &params, sizeof(ReduceParams));

    // Create the output tensor by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Set block and grid dimensions
    // Using 512 threads per block to maximize occupancy and ensure coalesced reads
    const int threads = 512;
    int blocks_y = (inner_size + threads - 1) / threads;
    dim3 grid(outer_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "const_aligned_max_reduce_forward", ([&] {
        combined_max_reduce_kernel<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &const_aligned_max_reduce_cuda_forward, "Combined constant and coalesced max reduction forward (CUDA)");
}
