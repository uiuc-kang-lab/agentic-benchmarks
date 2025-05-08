#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Store frequently accessed, read-only parameters in constant memory
__constant__ int64_t c_dim_size;
__constant__ int64_t c_inner_size;

// Kernel performing max reduction using constant memory for dim_size and inner_size
// The input tensor is conceptually shaped as [outer, dim, inner]
// The output tensor is shaped as [outer, inner]

template <typename scalar_t>
__global__ void const_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = outer_size * c_inner_size;
    if (idx >= total) return;

    // Compute outer and inner indices
    int outer_idx = idx / c_inner_size;
    int inner_idx = idx % c_inner_size;
    
    // Compute the base offset for the current outer index
    int64_t base = outer_idx * c_dim_size * c_inner_size;
    
    // Initialize maximum value using the first element along the reduction dimension
    scalar_t max_val = input[base + inner_idx];
    
    // Perform max reduction along the specified dimension
    for (int i = 1; i < c_dim_size; i++) {
        scalar_t val = input[base + i * c_inner_size + inner_idx];
        max_val = max(max_val, val);
    }
    
    // Write the reduced result to output
    output[idx] = max_val;
}


// Forward function that prepares the parameters, copies them to constant memory, and launches the kernel
torch::Tensor const_max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
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
    
    // Get the size along the reduction dimension
    int64_t dim_size = input.size(dim);
    
    // Copy frequently accessed, read-only parameters to constant memory
    cudaMemcpyToSymbol(c_dim_size, &dim_size, sizeof(int64_t));
    cudaMemcpyToSymbol(c_inner_size, &inner_size, sizeof(int64_t));
    
    // Prepare output tensor by removing the reduction dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    // Launch kernel
    const int threads = 256;
    int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "const_max_reduce_forward", ([&] {
        const_max_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &const_max_reduce_cuda_forward, "Constant memory based max reduction forward (CUDA)");
}
