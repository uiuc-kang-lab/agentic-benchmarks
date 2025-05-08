#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses a 2D grid mapping to correspond directly to outer and inner dimensions
// to better map threads to the work domain, leading to more efficient use of hardware threads.

template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t dim_size,
    int64_t inner_size,
    int64_t outer_size) {

    // 2D thread indexing: grid.x corresponds to outer dimension, grid.y corresponds to inner dimension
    int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int inner_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        // Compute the output index
        int64_t out_index = outer_idx * inner_size + inner_idx;
        // Starting offset for the reduction along the specified dimension
        int64_t offset = outer_idx * dim_size * inner_size + inner_idx;
        
        scalar_t sum = 0;
        for (int i = 0; i < dim_size; i++) {
            sum += __ldg(input + offset + i * inner_size);
        }
        output[out_index] = sum / static_cast<scalar_t>(dim_size);
    }
}

// CUDA wrapper function
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    // Calculate the sizes
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Create output tensor; the reduction dimension is removed
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    // Define a 2D block and grid that maps the outer and inner dimensions directly
    dim3 block(16, 16);
    dim3 grid((outer_size + block.x - 1) / block.x, (inner_size + block.y - 1) / block.y);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            inner_size,
            outer_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean reduction with 2D thread indexing (CUDA)");
}
