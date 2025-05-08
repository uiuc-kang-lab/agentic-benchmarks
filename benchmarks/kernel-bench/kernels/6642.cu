#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses a 2D grid: grid.x corresponds to the outer dimension and grid.y covers the inner dimension.
// Each thread computes the maximum along the reduction dimension for one output element.
// For a fixed outer index, threads in the warp read consecutive (coalesced) global memory locations for each step in the reduction.

template <typename scalar_t>
__global__ void coalesced_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Each block in grid.x corresponds to one outer slice
    int outer_idx = blockIdx.x;
    // blockIdx.y and threadIdx.x together cover the inner dimension indices
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (inner_idx >= inner_size) return;

    // Compute the starting index for reduction along the specified dimension
    const int64_t input_offset = outer_idx * dim_size * inner_size + inner_idx;
    const scalar_t* in_ptr = input + input_offset;
    scalar_t max_val = *in_ptr;
    in_ptr += inner_size;

    int i = 1;
    #pragma unroll 4
    for (; i <= dim_size - 4; i += 4) {
        scalar_t v1 = *in_ptr; in_ptr += inner_size;
        scalar_t v2 = *in_ptr; in_ptr += inner_size;
        scalar_t v3 = *in_ptr; in_ptr += inner_size;
        scalar_t v4 = *in_ptr; in_ptr += inner_size;
        max_val = max(max(max_val, v1), max(max(v2, v3), v4));
    }
    for (; i < dim_size; i++) {
        scalar_t v = *in_ptr;
        in_ptr += inner_size;
        max_val = max(max_val, v);
    }

    // Write the result to the output tensor
    output[outer_idx * inner_size + inner_idx] = max_val;
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Calculate the product of sizes for outer dimensions (before the reduction dimension)
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    // The size along the reduction dimension
    int64_t dim_size = input.size(dim);

    // Calculate the product of the sizes for inner dimensions (after the reduction dimension)
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    // Prepare output tensor with the reduction dimension removed
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Define block and grid dimensions to ensure coalesced accesses
    const int threads = 256;
    const int blocks_y = (inner_size + threads - 1) / threads;
    dim3 grid(outer_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        coalesced_max_reduce_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Max reduction forward (CUDA) with memory coalescing");
}
