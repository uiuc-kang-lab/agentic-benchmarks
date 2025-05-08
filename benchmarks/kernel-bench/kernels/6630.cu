#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses a stride loop to allow each thread to process multiple output indices
// in case the total number of output elements (outer_size * inner_size) exceeds the available threads.

template <typename scalar_t>
__global__ void stride_loop_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t inner_size,
    const int64_t dim_size
) {
    // Calculate total number of output elements
    int64_t total = outer_size * inner_size;
    
    // Each thread will jump in strides of (blockDim.x * gridDim.x)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process all output elements in a stride loop
    while (idx < total) {
        // Map the linear index to (outer_idx, inner_idx)
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;

        // Compute the starting offset for this outer index
        int64_t base_offset = outer_idx * dim_size * inner_size;

        // Initialize max reduction with the first element
        scalar_t max_val = input[base_offset + inner_idx];

        // Process the reduction dimension with correct boundary handling
        for (int i = 1; i < dim_size; i++) {
            scalar_t candidate = input[base_offset + i * inner_size + inner_idx];
            max_val = max(max_val, candidate);
        }

        // Write the result for this output element
        output[idx] = max_val;

        // Move to the next element assigned to this thread
        idx += stride;
    }
}

// Forward function that sets up the kernel launch, computes tensor dimensions and handles negative dim

torch::Tensor stride_loop_max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Compute outer_size as the product of dimensions before 'dim'
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    // Compute inner_size as the product of dimensions after 'dim'
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    // The size of the reduction dimension
    int64_t dim_size = input.size(dim);

    // Prepare the output tensor by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Total number of output elements
    int64_t total_output = outer_size * inner_size;

    // Configure kernel launch parameters
    const int threads = 256;
    // Launch enough blocks so that total_threads (blocks * threads) is less than or equal to total_output
    int blocks = (total_output + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "stride_loop_max_reduce_forward", ([&] {
        stride_loop_max_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            inner_size,
            dim_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &stride_loop_max_reduce_cuda_forward, "Stride loop Max reduction forward (CUDA)");
}
