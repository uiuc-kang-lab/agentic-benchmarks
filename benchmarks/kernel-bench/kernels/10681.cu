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
    int row = blockIdx.x;
    const int64_t row_offset = row * n;

    const int tid = threadIdx.x;
    const int lane = tid & 31;       // lane index within a warp
    const int warp_id = tid >> 5;    // warp index within the block

    scalar_t val = 0;
    if (tid < n) {
        val = input[row_offset + (n - 1 - tid)];
    }

    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t n_val = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += n_val;
        }
    }

    __shared__ scalar_t warp_sums[32];  // Maximum of 32 warps per block (1024 threads)
    __shared__ scalar_t warp_offsets[32];
    int block_warps = (n + 31) / 32;
    if ((lane == 31) || (tid == n - 1)) {
        warp_sums[warp_id] = val;
    }

    __syncthreads();

    if (tid == 0) {
        warp_offsets[0] = 0;
        for (int i = 1; i < block_warps; i++) {
            warp_offsets[i] = warp_offsets[i - 1] + warp_sums[i - 1];
        }
    }

    __syncthreads();

    if (tid < n && warp_id > 0) {
        val += warp_offsets[warp_id];
    }

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

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda_parallel", ([&] {
            reverse_cumsum_parallel_kernel<scalar_t><<<blocks, threadBlock>>>(
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
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum along a specified dimension efficiently (CUDA)");
}
