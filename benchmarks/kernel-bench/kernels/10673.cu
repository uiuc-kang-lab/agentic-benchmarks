#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Parallel warp-scan kernel for reverse cumulative sum along the last dimension.
// This kernel works for contiguous tensors along dimension (dim == ndim - 1) and when n <= 1024.
// Each block processes one row. The algorithm loads the row in reverse order, performs an inclusive scan using warp shuffle intrinsics,
// and then writes the result back in the correct order. Only two __syncthreads() calls are used (for cross-warp accumulation).

template <typename scalar_t>
__global__ void reverse_cumsum_parallel_kernel(const scalar_t* __restrict__ input,
                                                 scalar_t* __restrict__ output,
                                                 const int64_t n) {
    // Each block processes one row
    int row = blockIdx.x;
    const int64_t row_offset = row * n;

    const int tid = threadIdx.x;
    const int lane = tid & 31;       // lane index within a warp
    const int warp_id = tid >> 5;    // warp index within the block

    // Load element in reverse order if within bounds
    scalar_t val = 0;
    if (tid < n) {
        // Access reversed: thread tid loads element at index (n - 1 - tid)
        val = input[row_offset + (n - 1 - tid)];
    }

    // Intra-warp inclusive scan using warp shuffle
    // No __syncthreads() needed inside a warp
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t n_val = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += n_val;
        }
    }

    // Write each warp's total (last active thread in the warp) into shared memory
    __shared__ scalar_t warp_sums[32];  // Maximum of 32 warps per block (1024 threads)
    __shared__ scalar_t warp_offsets[32];
    int block_warps = (n + 31) / 32;
    if ((lane == 31) || (tid == n - 1)) {
        warp_sums[warp_id] = val;
    }

    __syncthreads();

    // Compute the prefix sum of warp sums to get each warp's offset
    if (tid == 0) {
        warp_offsets[0] = 0;
        for (int i = 1; i < block_warps; i++) {
            warp_offsets[i] = warp_offsets[i - 1] + warp_sums[i - 1];
        }
    }

    __syncthreads();

    // Add the warp's offset to each thread's scan value
    if (tid < n && warp_id > 0) {
        val += warp_offsets[warp_id];
    }

    // Write the result back in reversed order to produce the reverse cumulative sum
    if (tid < n) {
        output[row_offset + (n - 1 - tid)] = val;
    }
}


// Sequential kernel fallback for non-contiguous cases or when n > 1024.
// This kernel uses a grid-stride loop to process each outer slice and does a sequential reverse cumulative sum.

template <typename scalar_t>
__global__ void reverse_cumsum_sequential_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer,
    const int64_t n,
    const int ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ strides,
    const int dim) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t gridStride = blockDim.x * gridDim.x;
    
    for (int64_t r = idx; r < outer; r += gridStride) {
        int64_t offset = 0;
        if (dim == ndim - 1) {
            offset = r * n;
        } else {
            int64_t tmp = r;
            for (int d = ndim - 1; d >= 0; d--) {
                if (d == dim) continue;
                int64_t cur_size = sizes[d];
                int64_t idx_d = tmp % cur_size;
                tmp /= cur_size;
                offset += idx_d * strides[d];
            }
        }
        int64_t stride_dim = (dim == ndim - 1) ? 1 : strides[dim];
        scalar_t cum = scalar_t(0);
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
    int64_t outer = x.numel() / n;

    // Fast path: if the cumulative sum is along the last dimension and n <= 1024, use the parallel warp-scan kernel
    if (dim == ndim - 1 && n <= 1024) {
        // Determine block size as next power-of-2 >= n, up to 1024
        int threads = 1;
        while (threads < n) {
            threads *= 2;
        }
        if (threads > 1024) threads = 1024;

        // Each block processes one row
        dim3 blocks(outer);
        dim3 threadBlock(threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda_parallel", ([&] {
            reverse_cumsum_parallel_kernel<scalar_t><<<blocks, threadBlock>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n);
        }));
    } else {
        // Fallback to sequential kernel
        // Prepare device arrays for sizes and strides
        const int max_dims = 10;
        int64_t h_sizes[max_dims], h_strides[max_dims];
        for (int i = 0; i < ndim; i++) {
            h_sizes[i] = x.size(i);
            h_strides[i] = x.stride(i);
        }

        int64_t *d_sizes = nullptr, *d_strides = nullptr;
        cudaError_t err = cudaMalloc(&d_sizes, ndim * sizeof(int64_t));
        TORCH_CHECK(err == cudaSuccess, "cudaMalloc for d_sizes failed");
        err = cudaMalloc(&d_strides, ndim * sizeof(int64_t));
        TORCH_CHECK(err == cudaSuccess, "cudaMalloc for d_strides failed");

        err = cudaMemcpy(d_sizes, h_sizes, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpy for d_sizes failed");
        err = cudaMemcpy(d_strides, h_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpy for d_strides failed");

        const int threads = 256;
        const int blocks = (outer + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda_sequential", ([&] {
            reverse_cumsum_sequential_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer,
                n,
                ndim,
                d_sizes,
                d_strides,
                dim);
        }));

        cudaFree(d_sizes);
        cudaFree(d_strides);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with parallel warp-scan for contiguous tensors (CUDA)");
}
