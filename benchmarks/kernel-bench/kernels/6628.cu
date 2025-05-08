#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel using __ldg for read-only memory access and aligning memory accesses

template <typename scalar_t>
__global__ void ldg_max_reduce_kernel(
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
    scalar_t max_val = __ldg(&input[base_offset + inner_idx]);

    // Loop over the reduction dimension
    for (int i = 1; i < dim_size; i++) {
        scalar_t val = __ldg(&input[base_offset + i * inner_size + inner_idx]);
        max_val = max(max_val, val);
    }

    // Write the result to output
    output[outer_idx * inner_size + inner_idx] = max_val;
}

// Function to launch the optimized kernel
torch::Tensor ldg_max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Handle negative dimension
    if (dim < 0) dim += input.dim();

    // Calculate sizes
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }
    
    const int64_t dim_size = input.size(dim);

    // Create output tensor
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Configure block and grid sizes
    const int threads = 256;
    int blocks_y = (inner_size + threads - 1) / threads;
    dim3 grid(outer_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "ldg_max_reduce_forward", ([&] {
        ldg_max_reduce_kernel<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            inner_size,
            dim_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ldg_max_reduce_cuda_forward, "LDG Max reduction forward (CUDA)");
}