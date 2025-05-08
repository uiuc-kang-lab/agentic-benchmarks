#include <torch/extension.h>
#include <vector>
#include <numeric>  // for iota

// Helper function to compute the next power of 2 for a given integer.
inline int nextPow2(int x) {
    int power = 1;
    while (power < x) {
        power *= 2;
    }
    return power;
}

// This kernel computes the reverse cumulative sum for each row (slice) of the input.
// It first loads the row in reverse order into shared memory, performs an inclusive scan
// (using the Hillis-Steele algorithm) on the reversed data, then writes the scanned
// results back in reverse order to produce the correct reverse cumulative sum.

template <typename scalar_t>
__global__ void reverse_cumsum_scan_kernel(const scalar_t* __restrict__ input,
                                             scalar_t* __restrict__ output,
                                             int row_size) {
    int row = blockIdx.x;  // each block processes one row
    int n = row_size;
    int idx = threadIdx.x;
    
    // Allocate shared memory for double buffering (temp and scan arrays).
    extern __shared__ char smem[];
    scalar_t* temp = reinterpret_cast<scalar_t*>(smem);
    scalar_t* scan = temp + blockDim.x;
    
    int row_offset = row * row_size;
    
    // Load the row in reversed order into shared memory.
    // If idx is outside the valid range, initialize to 0.
    if (idx < n) {
        temp[idx] = input[row_offset + (n - 1 - idx)];
    } else {
        temp[idx] = static_cast<scalar_t>(0);
    }
    __syncthreads();

    // Perform an inclusive scan using a Hillis-Steele algorithm.
    // This loop runs in O(log(blockDim.x)) steps.
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        scalar_t add_val = static_cast<scalar_t>(0);
        if (idx >= offset)
            add_val = temp[idx - offset];
        __syncthreads();
        scan[idx] = temp[idx] + add_val;
        __syncthreads();
        temp[idx] = scan[idx];
        __syncthreads();
    }
    
    // Write the result back to global memory in reversed order so that
    // each element receives the sum of itself and all subsequent elements.
    if (idx < n) {
        output[row_offset + (n - 1 - idx)] = temp[idx];
    }
}

// Host function for the reverse cumulative sum operation.
// This function ensures the input is contiguous and (if necessary) permuted so that
// the cumulative sum dimension is the last dimension. It then launches one block per row,
// with the block size chosen as the next power of two for the row length, ensuring
// even workload distribution among threads and blocks.

at::Tensor reverse_cumsum_optimized(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    
    // Permute tensor if the cumulative sum dimension is not the last dimension
    bool permuted = false;
    std::vector<int64_t> permute_dims(x.dim());
    std::iota(permute_dims.begin(), permute_dims.end(), 0);
    if (dim != x.dim() - 1) {
        std::swap(permute_dims[dim], permute_dims.back());
        x = x.permute(permute_dims).contiguous();
        permuted = true;
        dim = x.dim() - 1;
    }
    
    int64_t row_size = x.size(dim);
    int64_t num_rows = x.numel() / row_size;
    auto output = torch::empty_like(x);
    
    // Determine block size as the next power of two for the row size
    int block_size = min(nextPow2(static_cast<int>(row_size)), 1024);
    // Allocate dynamic shared memory: 2 arrays of block_size elements each
    size_t shared_mem_size = 2 * block_size * sizeof(float);  // type will be updated in the dispatch
    
    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "reverse_cumsum_parallel_scan", ([&] {
        using scalar = scalar_t; 
        shared_mem_size = 2 * block_size * sizeof(scalar);
        dim3 blocks(num_rows);
        dim3 threads(block_size);
        reverse_cumsum_scan_kernel<scalar><<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<scalar>(),
            output.data_ptr<scalar>(),
            row_size
        );
    }));
    
    // If we permuted earlier, undo the permutation to restore the original tensor shape
    if (permuted) {
        std::swap(permute_dims[dim], permute_dims.back());
        output = output.permute(permute_dims).contiguous();
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum_optimized, "Optimized reverse cumulative sum using parallel scan");
}
