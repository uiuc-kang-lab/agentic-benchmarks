#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int c_reverse_dim;

template<typename scalar_t>
__global__ void reverse_cumsum_kernel(const scalar_t* input, scalar_t* output, int64_t dim_size, int64_t total_elements) {
    int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;
    
    const int64_t stride = total_elements / dim_size;
    const int64_t pos_in_chunk = gid % stride;
    const int64_t chunk_id = gid / stride;
    
    scalar_t sum = 0;
    for (int i = c_reverse_dim-1; i >= 0; --i) {
        int64_t element_idx = chunk_id * c_reverse_dim * stride + i * stride + pos_in_chunk;
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