#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to compute reverse cumulative sum using shared memory for efficiency.
// This kernel is optimized for contiguous tensors along the last dimension and n <= 1024.

template <typename scalar_t>
__global__ void reverse_cumsum_shared_kernel(const scalar_t* __restrict__ input,
                                              scalar_t* __restrict__ output,
                                              const int64_t n) {
    extern __shared__ scalar_t sdata[];  // Shared memory for warp sums
    int row = blockIdx.x;
    const int64_t row_offset = row * n;

    const int tid = threadIdx.x;
    const int lane = tid & 31;       // lane index within a warp
    const int warp_id = tid >> 5;    // warp index within the block

    scalar_t val = 0;
    if (tid < n) {
        val = input[row_offset + (n - 1 - tid)];
    }

    // Load into shared memory
    sdata[tid] = val;
    __syncthreads();

    // Intra-warp inclusive scan using warp shuffle
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t n_val = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += n_val;
        }
    }

    // Write each warp's total (last active thread in the warp) into shared memory
    if ((lane == 31) || (tid == n - 1)) {
        sdata[warp_id * 32 + lane] = val;
    }

    __syncthreads();

    // Compute the prefix sum of warp sums to get each warp's offset
    if (tid < 32) {  // Only first warp computes
        scalar_t warp_sum = 0;
        for (int i = 0; i < warp_id; i++) {
            warp_sum += sdata[i * 32 + 31];
        }
        sdata[tid] = warp_sum;
    }

    __syncthreads();

    // Add the warp's offset to each thread's scan value
    if (tid < n && warp_id > 0) {
        val += sdata[warp_id];
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

        size_t shared_mem_size = threads * sizeof(scalar_t);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda_shared", ([&] {
            reverse_cumsum_shared_kernel<scalar_t><<<blocks, threadBlock, shared_mem_size>>>(
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
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum using shared memory (CUDA)");
}
