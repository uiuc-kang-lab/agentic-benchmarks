#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_reduce_kernel(
    const scalar_t* input,
    scalar_t* output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = gridDim.x;
    
    // Calculate the number of elements per thread block
    const int elements_per_block = (outer_size * inner_size + grid_size - 1) / grid_size;
    const int block_start = bid * elements_per_block;
    const int block_end = min(block_start + elements_per_block, static_cast<int>(outer_size * inner_size));
    
    for (int idx = block_start + tid; idx < block_end; idx += block_size) {
        const int outer_idx = idx / inner_size;
        const int inner_idx = idx % inner_size;
        const int64_t start_idx = outer_idx * dim_size * inner_size + inner_idx;
        
        // Initialize with first element
        scalar_t max_val = input[start_idx];
        
        // Reduce along dimension
        #pragma unroll
        for (int i = 1; i < dim_size; i++) {
            const scalar_t val = input[start_idx + i * inner_size];
            max_val = max(max_val, val);
        }
        
        output[idx] = max_val;
    }
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
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
    
    const int threads = 256;
    const int blocks = min(256, static_cast<int>((outer_size * inner_size + threads - 1) / threads));
    const int shared_mem_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        max_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA)");
}