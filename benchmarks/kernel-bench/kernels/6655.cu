#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel with optimized block size for better performance on specific hardware

template <typename scalar_t>
__global__ void optimized_block_size_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Compute the index for each thread in global space
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (inner_idx >= inner_size) return; // Ensure threads do not access invalid memory

    // Each block handles a slice in outer dimension ensuring coalesced access
    int outer_idx = blockIdx.x;
    const int64_t input_offset = outer_idx * dim_size * inner_size + inner_idx;

    scalar_t max_val = input[input_offset];

    // Loop over the reduction dimension; threads access coalesced positions
    for (int i = 1; i < dim_size; i++) {
        scalar_t val = input[input_offset + i * inner_size];
        max_val = fmaxf(max_val, val);
    }

    // Store result in output
    output[outer_idx * inner_size + inner_idx] = max_val;
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    const int64_t dim_size = input.size(dim);

    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const int threads = 512;  // Experimented optimal block size
    const int blocks_y = (inner_size + threads - 1) / threads;
    dim3 grid(outer_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        optimized_block_size_max_reduce_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Optimized Block Size Max Reduction Forward (CUDA)");
}