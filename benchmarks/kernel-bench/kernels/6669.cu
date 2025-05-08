#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel implements max reduction along a given dimension with loop unrolling
// to reduce loop overhead, while ensuring that global memory accesses are coalesced.
// Each thread is responsible for computing the maximum over the reduction dimension
// for a fixed combination of outer and inner indices.

template <typename scalar_t>
__global__ void unrolled_coalesced_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Compute the index along the inner dimension
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (inner_idx >= inner_size) return;

    // Each block in grid.x corresponds to one outer slice
    int outer_idx = blockIdx.x;
    
    // Calculate base offset for this (outer_idx, inner_idx) pair
    const int64_t input_offset = outer_idx * dim_size * inner_size + inner_idx;

    // Initialize the maximum value with the first element
    scalar_t max_val = input[input_offset];

    // Unroll the loop with a factor of 4 to reduce loop overhead
    const int unroll = 4;
    int limit = (dim_size / unroll) * unroll;
    
    // Loop starting from 1 since we already used the 0th element
    for (int i = 1; i < limit; i += unroll) {
        int64_t base = input_offset + i * inner_size;
        scalar_t a = input[base];
        scalar_t b = input[base + inner_size];
        scalar_t c = input[base + 2LL * inner_size];
        scalar_t d = input[base + 3LL * inner_size];
        max_val = max(max_val, a);
        max_val = max(max_val, b);
        max_val = max(max_val, c);
        max_val = max(max_val, d);
    }

    // Handle remainder if dim_size is not divisible by unroll factor
    for (int i = limit; i < dim_size; i++) {
        scalar_t temp = input[input_offset + i * inner_size];
        max_val = max(max_val, temp);
    }

    // Write the computed maximum to the output tensor
    output[outer_idx * inner_size + inner_idx] = max_val;
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Compute the outer and inner sizes
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    const int64_t dim_size = input.size(dim);

    // Prepare the output tensor by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Choose block and grid sizes to ensure coalesced accesses
    const int threads = 256;
    const int blocks_y = (inner_size + threads - 1) / threads;
    dim3 grid(outer_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        unrolled_coalesced_max_reduce_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Unrolled Coalesced Max Reduction Forward (CUDA)");
}
