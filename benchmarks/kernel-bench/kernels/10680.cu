#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_DIMS 10

// Declare constant memory for sizes and strides (used in non-contiguous kernel)
__constant__ int64_t const_sizes[MAX_DIMS];
__constant__ int64_t const_strides[MAX_DIMS];

// Kernel for the contiguous case (dim == last dimension) using warp-scan with minimal __syncthreads
template <typename scalar_t>
__global__ void reverse_cumsum_parallel_kernel(const scalar_t* __restrict__ input,
                                                 scalar_t* __restrict__ output,
                                                 const int64_t n) {
    // Each block handles one row
    int row = blockIdx.x;
    int64_t row_offset = row * n;
    int tid = threadIdx.x;
    int lane = tid & 31;      // lane index within a warp
    int warp_id = tid >> 5;   // warp index within the block

    scalar_t val = 0;
    if (tid < n) {
        // Load element in reverse order
        val = input[row_offset + (n - 1 - tid)];
    }

    // Intra-warp inclusive scan using warp shuffle (no __syncthreads needed within a warp)
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t y = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset)
            val += y;
    }

    // Shared memory to hold the last value of each warp
    __shared__ scalar_t warp_sums[32];
    int num_warps = (n + 31) / 32;
    if ((lane == 31) || (tid == n - 1)) {
        warp_sums[warp_id] = val;
    }

    __syncthreads(); // Ensure all warp sums are written

    // Thread 0 computes prefix sums for warp offsets
    if (tid == 0) {
        for (int i = 1; i < num_warps; i++) {
            warp_sums[i] += warp_sums[i - 1];
        }
    }

    __syncthreads(); // Ensure warp offsets are available

    // Each thread (except in warp 0) adds the offset from previous warps
    if (tid < n && warp_id > 0) {
        val += warp_sums[warp_id - 1];
    }

    // Write the result back in the proper order to produce the reverse cumulative sum
    if (tid < n) {
        output[row_offset + (n - 1 - tid)] = val;
    }
}

// Kernel for non-contiguous case (or when the cumsum dimension is not the last)
// Uses grid-stride loops and constant memory for sizes and strides to avoid dynamic allocation.

template <typename scalar_t>
__global__ void reverse_cumsum_noncontiguous_kernel(const scalar_t* __restrict__ input,
                                                       scalar_t* __restrict__ output,
                                                       const int64_t outer,
                                                       const int64_t n,
                                                       const int ndim,
                                                       const int dim) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t gridStride = blockDim.x * gridDim.x;
    for (int64_t r = idx; r < outer; r += gridStride) {
        int64_t offset = 0;
        int64_t tmp = r;
        // Decompose the linear index 'r' into multi-dimensional indices (skipping the cumsum dimension)
        for (int d = ndim - 1; d >= 0; d--) {
            if (d == dim) continue;
            int64_t cur_size = const_sizes[d];
            int64_t idx_d = tmp % cur_size;
            tmp /= cur_size;
            offset += idx_d * const_strides[d];
        }
        // Determine stride for the cumsum dimension
        int64_t stride_dim = (dim == ndim - 1) ? 1 : const_strides[dim];
        scalar_t cum = scalar_t(0);
        // Calculate reverse cumulative sum along the cumsum dimension
        for (int64_t j = n - 1; j >= 0; j--) {
            int64_t cur_index = offset + j * stride_dim;
            cum += input[cur_index];
            output[cur_index] = cum;
        }
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    // Ensure the tensor is contiguous and on CUDA
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");

    const int ndim = x.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    auto output = at::empty_like(x);
    int64_t n = x.size(dim);
    int64_t total = x.numel();
    int64_t outer = total / n;

    if (dim == ndim - 1) {
        // For the contiguous (last dimension) case, use the optimized warp-scan kernel
        dim3 blocks(outer);
        // Use min(n, 1024) threads per block
        int threads = (n < 1024 ? n : 1024);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda_parallel", ([&] {
            reverse_cumsum_parallel_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n);
        }));
    } else {
        // For non-contiguous cases, use the grid-stride loop kernel with constant memory for sizes/strides
        int64_t h_sizes[MAX_DIMS], h_strides[MAX_DIMS];
        for (int i = 0; i < ndim; i++) {
            h_sizes[i] = x.size(i);
            h_strides[i] = x.stride(i);
        }
        // Copy sizes and strides to constant memory to avoid cudaMalloc/free overhead
        cudaMemcpyToSymbol(const_sizes, h_sizes, ndim * sizeof(int64_t));
        cudaMemcpyToSymbol(const_strides, h_strides, ndim * sizeof(int64_t));

        int threads = 256;
        int blocks = (outer + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda_noncontiguous", ([&] {
            reverse_cumsum_noncontiguous_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer,
                n,
                ndim,
                dim);
        }));
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum optimized with minimal synchronizations (CUDA)");
}
