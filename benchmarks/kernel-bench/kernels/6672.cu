#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use constant memory for frequently accessed read-only data
__constant__ int64_t const_outer_size;
__constant__ int64_t const_dim_size;
__constant__ int64_t const_inner_size;

// Kernel with optimized block size and constant memory usage

template <typename scalar_t>
__global__ void constant_memory_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output
) {
    // Compute the index for each thread in global space
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (inner_idx >= const_inner_size) return; // Ensure threads do not access invalid memory

    // Each block handles a slice in outer dimension ensuring coalesced access
    int outer_idx = blockIdx.x;
    const int64_t input_offset = outer_idx * const_dim_size * const_inner_size + inner_idx;

    scalar_t max_val = input[input_offset];

    // Loop over the reduction dimension; threads access coalesced positions
    for (int i = 1; i < const_dim_size; i++) {
        scalar_t val = input[input_offset + i * const_inner_size];
        max_val = max(max_val, val);
    }

    // Store result in output
    output[outer_idx * const_inner_size + inner_idx] = max_val;
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

    // Copy sizes to constant memory
    cudaMemcpyToSymbolAsync(const_outer_size, &outer_size, sizeof(int64_t), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(const_dim_size, &dim_size, sizeof(int64_t));
    cudaMemcpyToSymbol(const_inner_size, &inner_size, sizeof(int64_t));

    const int threads = 512;  // Experimented optimal block size
    const int blocks_y = (inner_size + threads - 1) / threads;
    dim3 grid(outer_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        constant_memory_max_reduce_kernel<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Constant Memory Max Reduction Forward (CUDA)");
}