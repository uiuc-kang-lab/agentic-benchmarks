#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for dimension size
__constant__ int64_t c_dim_size;

template <typename scalar_t>
__global__ void max_reduce_kernel_optimized(
    const scalar_t* input,
    scalar_t* output,
    const int64_t outer_size,
    const int64_t inner_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = outer_size * inner_size;
    
    if (idx >= total_elements) return;
    
    const int outer_idx = idx / inner_size;
    const int inner_idx = idx % inner_size;
    
    // Calculate starting position for this thread
    const int64_t start_idx = outer_idx * c_dim_size * inner_size + inner_idx;
    
    // Initialize with first element
    scalar_t max_val = input[start_idx];
    
    // Reduce along dimension
    for (int i = 1; i < c_dim_size; i++) {
        const scalar_t val = input[start_idx + i * inner_size];
        max_val = max(max_val, val);
    }
    
    output[idx] = max_val;
}

torch::Tensor max_reduce_cuda_forward_optimized(torch::Tensor input, int64_t dim) {
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
    
    // Copy dim_size to constant memory
    cudaMemcpyToSymbol(c_dim_size, &dim_size, sizeof(int64_t));
    
    // Create output tensor
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward_optimized", ([&] {
        max_reduce_kernel_optimized<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            inner_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward_optimized, "Max reduce forward optimized (CUDA)");
}