#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_reduce_kernel_optimized(
    const scalar_t* input,
    scalar_t* output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    extern __shared__ scalar_t shared_data[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = outer_size * inner_size;
    
    if (idx >= total_elements) return;
    
    const int outer_idx = idx / inner_size;
    const int inner_idx = idx % inner_size;
    
    // Calculate starting position for this thread
    const int64_t start_idx = outer_idx * dim_size * inner_size + inner_idx;
    
    // Initialize with first element
    scalar_t max_val = input[start_idx];
    
    // Reduce along dimension
    for (int i = 1; i < dim_size; i++) {
        const scalar_t val = input[start_idx + i * inner_size];
        max_val = max(max_val, val);
    }
    
    // Store the result in shared memory
    shared_data[threadIdx.x] = max_val;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] = max(shared_data[threadIdx.x], shared_data[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared_data[0];
    }
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
    
    // Create output tensor
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward_optimized", ([&] {
        const int shared_mem_size = threads * sizeof(scalar_t);
        max_reduce_kernel_optimized<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &max_reduce_cuda_forward_optimized, "Max reduce forward optimized (CUDA)");
}