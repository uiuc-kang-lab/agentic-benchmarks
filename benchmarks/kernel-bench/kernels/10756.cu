#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Removed constant memory usage for dimension size

template<typename scalar_t>
__global__ void reverse_cumsum_kernel(const scalar_t* input, scalar_t* output, int64_t dim_size, int64_t inner_count, int64_t total_lines) {
    int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_lines) return;
    
    // Compute position within the inner (contiguous) dimension and outer index
    int64_t pos_in_line = gid % inner_count;
    int64_t outer_id = gid / inner_count;
    
    scalar_t sum = 0;
    // Iterate over the target dimension in reverse order
    for (int i = dim_size - 1; i >= 0; --i) {
        int64_t element_idx = outer_id * (dim_size * inner_count) + i * inner_count + pos_in_line;
        sum += input[element_idx];
        output[element_idx] = sum;
    }
}

at::Tensor reverse_cumsum_optimized(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    
    const int64_t dim_size = x.size(dim);
    cudaMemcpyToSymbol(c_reverse_dim, &dim_size, sizeof(int64_t));
    
    auto output = torch::empty_like(x);
    const int64_t num_blocks = 256;
    const int64_t threads_per_block = 256;
    const int64_t total_elements = x.numel();
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum", [&] {
        reverse_cumsum_kernel<scalar_t><<<(total_elements + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            total_elements
        );
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum_optimized, "Optimized reverse cumulative sum with constant memory");
}