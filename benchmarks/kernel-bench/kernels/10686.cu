#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Parallel warp-scan kernel for reverse cumulative sum along the last dimension.
// This kernel uses shared memory to reduce global memory access latency.
// Each block processes one row, loading elements into shared memory for reuse.

template <typename scalar_t>
__global__ void reverse_cumsum_shared_kernel(const scalar_t* __restrict__ input,
                                             scalar_t* __restrict__ output,
                                             const int64_t n) {
    extern __shared__ scalar_t shared_mem[];
    int row = blockIdx.x;
    const int64_t row_offset = row * n;

    const int tid = threadIdx.x;
    const int lane = tid & 31;       // lane index within a warp
    const int warp_id = tid >> 5;    // warp index within the block

    // Load elements into shared memory in reverse order
    if (tid < n) {
        shared_mem[tid] = input[row_offset + (n - 1 - tid)];
    }
    __syncthreads();

    // Perform warp-scan using shared memory
    scalar_t val = (tid < n) ? shared_mem[tid] : 0;
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t n_val = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += n_val;
        }
    }

    // Store the warp sums in shared memory
    __shared__ scalar_t warp_sums[32];
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

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    // Ensure the tensor is contiguous and on CUDA
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");

    const int ndim = x.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    auto output = at::empty_like(x);

    int64_t n = x.size(dim);
    int64_t outer = x.numel() / n;

    if (dim == ndim - 1 && n <= 1024) {
        int threads = 1;
        while (threads < n) {
            threads *= 2;
        }
        if (threads > 1024) threads = 1024;

        dim3 blocks(outer);
        dim3 threadBlock(threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda_shared", ([&] {
            reverse_cumsum_shared_kernel<scalar_t><<<blocks, threadBlock, threads * sizeof(scalar_t)>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n);
        }));
    } else {
        // For other cases, leverage the flip and cumsum operations for simplicity and device efficiency.
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        output = cumsum.flip(dim);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with shared memory optimization (CUDA)");
}
