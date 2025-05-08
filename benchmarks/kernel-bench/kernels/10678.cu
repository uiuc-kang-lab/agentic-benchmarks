#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void reverse_cumsum_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t n,
    const int64_t outer_size) {
    
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    
    // Each block handles one row
    const int64_t row = bid;
    if (row >= outer_size) return;
    
    const int64_t row_offset = row * n;
    
    // Load data into shared memory in reverse order
    for (int i = tid; i < n; i += block_size) {
        shared_data[i] = input[row_offset + (n - 1 - i)];
    }
    __syncthreads();
    
    // Compute prefix sum in shared memory
    for (int i = tid; i < n; i += block_size) {
        scalar_t sum = 0;
        for (int j = 0; j <= i; j++) {
            sum += shared_data[j];
        }
        // Write back to global memory in correct order
        output[row_offset + (n - 1 - i)] = sum;
    }
}

template <typename scalar_t>
__global__ void reverse_cumsum_noncontiguous_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t inner_size,
    const int64_t stride) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    
    // Process multiple rows per block using grid-stride loop
    for (int64_t row = bid; row < outer_size; row += gridDim.x) {
        const int64_t base_idx = row * stride;
        
        // Process elements within the row
        for (int i = tid; i < inner_size; i += block_size) {
            scalar_t sum = 0;
            const int64_t end_idx = base_idx + (inner_size - 1) * stride;
            const int64_t start_idx = base_idx + i * stride;
            
            // Compute reverse cumsum for this position
            for (int64_t j = end_idx; j >= start_idx; j -= stride) {
                sum += input[j];
            }
            output[start_idx] = sum;
        }
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    
    const int ndim = x.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");
    
    auto output = at::empty_like(x);
    const int64_t n = x.size(dim);
    const int64_t outer_size = x.numel() / n;
    
    // Use different kernels for contiguous (last dim) vs non-contiguous cases
    if (dim == ndim - 1) {
        // For last dimension, use shared memory approach
        const int block_size = 256;
        const int num_blocks = outer_size;
        const size_t shared_mem_size = n * sizeof(typename std::conditional<std::is_same<float, scalar_t>::value, float, double>::type);
        
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda", ([&] {
            reverse_cumsum_shared_kernel<scalar_t><<<num_blocks, block_size, shared_mem_size>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n,
                outer_size);
        }));
    } else {
        // For non-contiguous case
        const int block_size = 256;
        const int num_blocks = std::min(outer_size, static_cast<int64_t>(65535));
        
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda", ([&] {
            reverse_cumsum_noncontiguous_kernel<scalar_t><<<num_blocks, block_size>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                n,
                x.stride(dim));
        }));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with shared memory optimization (CUDA)");
}