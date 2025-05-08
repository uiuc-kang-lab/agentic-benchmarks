#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int UNROLL_FACTOR = 4>
__global__ void reverse_cumsum_unrolled_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t n,
    const int64_t stride) {
    
    // Calculate base indices
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t row = tid / ((n + UNROLL_FACTOR - 1) / UNROLL_FACTOR);
    const int64_t col_base = (tid % ((n + UNROLL_FACTOR - 1) / UNROLL_FACTOR)) * UNROLL_FACTOR;
    const int64_t row_offset = row * stride;

    // Load and process UNROLL_FACTOR elements at once
    scalar_t partial_sum = 0;
    
    // Process elements in reverse order with manual unrolling
    #pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        const int64_t col = col_base + i;
        if (col < n) {
            const int64_t rev_idx = n - 1 - col;
            partial_sum += input[row_offset + rev_idx];
            output[row_offset + rev_idx] = partial_sum;
        }
    }
}

template <typename scalar_t>
__global__ void reverse_cumsum_generic_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer,
    const int64_t inner_size,
    const int64_t stride) {
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t row = tid / 32;  // Use warp-sized groups
    
    if (row < outer) {
        const int64_t row_offset = row * stride;
        scalar_t sum = 0;
        
        // Process elements in reverse order
        #pragma unroll 4
        for (int64_t i = inner_size - 1; i >= 0; --i) {
            sum += input[row_offset + i];
            output[row_offset + i] = sum;
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
    const int64_t outer = x.numel() / n;

    // Fast path for last dimension
    if (dim == ndim - 1) {
        constexpr int THREADS_PER_BLOCK = 256;
        constexpr int UNROLL_FACTOR = 4;
        
        const int64_t elements_per_thread = UNROLL_FACTOR;
        const int64_t total_threads = (outer * ((n + UNROLL_FACTOR - 1) / UNROLL_FACTOR));
        const int64_t num_blocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda", ([&] {
            reverse_cumsum_unrolled_kernel<scalar_t, UNROLL_FACTOR>
                <<<num_blocks, THREADS_PER_BLOCK>>>(
                    x.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    n,
                    n);
        }));
    } else {
        // Generic case for non-last dimensions
        constexpr int THREADS_PER_BLOCK = 256;
        const int64_t stride = x.stride(dim);
        const int64_t num_blocks = (outer * 32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda", ([&] {
            reverse_cumsum_generic_kernel<scalar_t>
                <<<num_blocks, THREADS_PER_BLOCK>>>(
                    x.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    outer,
                    n,
                    stride);
        }));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with unrolled processing (CUDA)");
}