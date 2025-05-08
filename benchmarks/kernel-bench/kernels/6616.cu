#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Structure to hold frequently accessed read-only parameters
struct ReduceParams {
    int64_t dim_size;
    int64_t inner_size;
};

// Declare constant memory for the reduce parameters
__constant__ ReduceParams c_params;

// Kernel to perform max reduction over one dimension using constant memory for parameters
template <typename scalar_t>
__global__ void constdata_stride_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size
) {
    // Each block.x handles one slice from the outer dimensions
    int outer_idx = blockIdx.x;

    // Each thread computes for a specific inner dimension index, using grid.y and threadIdx.x
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

    // Use constant memory for inner_size
    if (inner_idx >= c_params.inner_size) return;

    // Base offset for this outer slice
    int64_t base_offset = outer_idx * c_params.dim_size * c_params.inner_size;

    // Initialize the max value using the first element along the reduction dimension
    scalar_t max_val = input[base_offset + inner_idx];

    // Loop over the reduction dimension (stored in constant memory)
    for (int i = 1; i < c_params.dim_size; i++) {
        scalar_t val = input[base_offset + i * c_params.inner_size + inner_idx];
        max_val = max(max_val, val);
    }

    // Write the result to the output tensor, which has dimensions [outer, inner]
    output[outer_idx * c_params.inner_size + inner_idx] = max_val;
}

// Host function that sets up the kernel launch
// It calculates sizes, copies frequently accessed parameters to constant memory,
// and launches a 2D grid where grid.x = outer size and grid.y covers the inner dimension

torch::Tensor constdata_stride_max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Adjust negative dimension
    if (dim < 0) dim += input.dim();

    // Compute outer_size: product of dimensions before the reduced dimension
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    // Compute inner_size: product of dimensions after the reduced dimension
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    // Get the size along the reduction dimension
    const int64_t dim_size = input.size(dim);

    // Prepare and copy the reduction parameters to constant memory
    ReduceParams params;
    params.dim_size = dim_size;
    params.inner_size = inner_size;
    cudaMemcpyToSymbol(c_params, &params, sizeof(ReduceParams));

    // Create the output tensor by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Configure the 2D grid: grid.x handles the outer dimension,
    // grid.y divides the inner dimension among threads
    const int threads = 256;
    int blocks_y = (inner_size + threads - 1) / threads;
    dim3 grid(outer_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "constdata_stride_max_reduce_forward", ([&] {
        constdata_stride_max_reduce_kernel<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &constdata_stride_max_reduce_cuda_forward, "Const Data Stride Max reduction forward (CUDA)");
}
