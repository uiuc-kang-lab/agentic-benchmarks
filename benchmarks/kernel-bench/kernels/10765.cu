#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel processes each slice (all dimensions except the cumsum axis) in one block.
// It breaks the slice into tiles (of size equal to blockDim.x) and processes them in reverse order.
// Within each tile, a shared-memory based Hillis-Steele inclusive scan (performed on the reversed tile) is used to compute
the reverse cumulative sum. The tile result is then written back in the proper order with an accumulated offset.
// Atomic operations are not used across blocks; synchronization is achieved through __syncthreads(), minimizing global memory contention.

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(const scalar_t* __restrict__ input,
                                        scalar_t* __restrict__ output,
                                        const int64_t dim_size,
                                        const int64_t stride,
                                        const int64_t num_slices) {
    int slice = blockIdx.x;
    if (slice >= num_slices)
        return;

    // Pointers to the beginning of the slice
    const scalar_t* slice_input = input + slice * dim_size * stride;
    scalar_t* slice_output = output + slice * dim_size * stride;

    int tile_size = blockDim.x; // number of threads per block
    int num_tiles = (dim_size + tile_size - 1) / tile_size;
    
    scalar_t offset = 0;  // running sum from tiles with higher indices
    
    extern __shared__ char smem[];
    scalar_t* s_data = reinterpret_cast<scalar_t*>(smem);

    // Process tiles in reverse order so that later (larger index) parts are computed first
    for (int tile = num_tiles - 1; tile >= 0; tile--) {
        int start = tile * tile_size;
        int m = (start + tile_size <= dim_size) ? tile_size : (dim_size - start);
        
        // Load current tile into shared memory in reversed order.
        // Each valid index i in [0, m) loads from global index: start + (m - 1 - i)
        for (int i = threadIdx.x; i < m; i += blockDim.x) {
            int global_index = start + (m - 1 - i);
            s_data[i] = slice_input[global_index * stride];
        }
        __syncthreads();
        
        // Perform an in-tile inclusive scan (Hillis-Steele) on s_data[0...m-1].
        // After this loop, for each index i, s_data[i] holds the sum of elements 0..i in the reversed tile.
        for (int d = 1; d < m; d *= 2) {
            scalar_t val = 0;
            if (threadIdx.x < m) {
                if (threadIdx.x >= d)
                    val = s_data[threadIdx.x - d];
            }
            __syncthreads();
            if (threadIdx.x < m) {
                s_data[threadIdx.x] += val;
            }
            __syncthreads();
        }
        
        // Write the scanned results back to global memory, adding the offset computed from later tiles.
        // Revert the order to place each result in its original location
        for (int i = threadIdx.x; i < m; i += blockDim.x) {
            int global_index = start + (m - 1 - i);
            slice_output[global_index * stride] = s_data[i] + offset;
        }
        __syncthreads();
        
        // Update the offset with the total sum of the current tile (last element in the scan)
        if (threadIdx.x == m - 1) {
            offset += s_data[m - 1];
        }
        __syncthreads();
    }
}

at::Tensor reverse_cumsum_optimized(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(x);
    
    int64_t dim_size = x.size(dim);
    int64_t num_slices = x.numel() / dim_size;
    int64_t stride = x.stride(dim);
    
    int threads = 256; // tile size; adjust as appropriate
    int blocks = num_slices;

    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "reverse_cumsum_optimized", ([&] {
        reverse_cumsum_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            stride,
            num_slices);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum_optimized, "Optimized reverse cumulative sum with block-level scan (CUDA)");
}
