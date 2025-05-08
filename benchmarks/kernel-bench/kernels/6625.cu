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

// Combined kernel to perform max reduction using constant memory and coalesced access
// Intermediate buffer for partial results when splitting large reductions
template <typename scalar_t>
__global__ void partial_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ partial_output,
    const int64_t outer_size,
    const int64_t chunk_size,
    const int64_t chunk_offset
) {
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (inner_idx >= c_params.inner_size) return;

    int64_t base_offset = outer_idx * c_params.dim_size * c_params.inner_size;
    
    // Start from this chunk's offset in the reduction dimension
    int64_t start_idx = chunk_offset;
    int64_t end_idx = min(start_idx + chunk_size, c_params.dim_size);
    
    // Initialize with first value in this chunk
    scalar_t max_val = input[base_offset + start_idx * c_params.inner_size + inner_idx];
    
    // Process this chunk of the reduction dimension
    for (int64_t i = start_idx + 1; i < end_idx; i++) {
        scalar_t val = input[base_offset + i * c_params.inner_size + inner_idx];
        max_val = max(max_val, val);
    }
    
    // Write partial result
    partial_output[(outer_idx * gridDim.z + blockIdx.z) * c_params.inner_size + inner_idx] = max_val;
}

// Final reduction kernel to combine partial results
template <typename scalar_t>
__global__ void final_max_reduce_kernel(
    const scalar_t* __restrict__ partial_results,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t num_chunks
) {
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (inner_idx >= c_params.inner_size) return;

    // Initialize with first partial result
    scalar_t max_val = partial_results[outer_idx * num_chunks * c_params.inner_size + inner_idx];
    
    // Combine all partial results
    for (int chunk = 1; chunk < num_chunks; chunk++) {
        scalar_t val = partial_results[(outer_idx * num_chunks + chunk) * c_params.inner_size + inner_idx];
        max_val = max(max_val, val);
    }
    
    // Write final result
    output[outer_idx * c_params.inner_size + inner_idx] = max_val;
}

// Host function that sets up the kernel launch
// It calculates sizes, copies frequently accessed parameters to constant memory,
// and launches a 2D grid where grid.x = outer size and grid.y covers the inner dimension

torch::Tensor optimized_coalesced_max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
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
    const int threads = 512;
    int blocks_y = (inner_size + threads - 1) / threads;
    dim3 grid(outer_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "optimized_coalesced_max_reduce_forward", ([&] {
        optimized_coalesced_max_reduce_kernel<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_coalesced_max_reduce_cuda_forward, "Optimized Coalesced Max reduction forward (CUDA)");
}
