#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes the reverse cumulative sum using a forward inclusive scan
// and then computing: out[i] = T - (prefix[i] - a), where T is the total sum of the row.
// Intra-warp inclusive scan is computed using __shfl_up_sync(), and the final total T is computed
// via a block reduction that leverages shared memory and __shfl_down_sync() for fast warp-level reduction.

template <typename scalar_t>
__global__ void shfl_reduce_reverse_cumsum_kernel(const scalar_t* __restrict__ input,
                                                    scalar_t* __restrict__ output,
                                                    int64_t n) {
    // Each block processes one row
    int row = blockIdx.x;
    const int base = row * n;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    // Load element from input (forward order)
    scalar_t a = (tid < n) ? input[base + tid] : 0;

    // Intra-warp inclusive scan using __shfl_up_sync()
    scalar_t prefix = a;
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t y = __shfl_up_sync(0xffffffff, prefix, offset);
        if (lane >= offset) {
            prefix += y;
        }
    }

    // Each warp's last thread (or the last valid thread) writes its warp total to shared memory
    __shared__ scalar_t warp_sums[32];
    if (tid < n && (lane == 31 || tid == n - 1)) {
        warp_sums[warp_id] = prefix;
    }
    __syncthreads();

    // Compute warp offsets via an exclusive scan on warp_sums (using a simple loop, num of warps is small)
    __shared__ scalar_t warp_offsets[32];
    int num_warps = (n + 31) / 32;
    if (warp_id == 0 && tid < num_warps) {
        scalar_t sum = 0;
        for (int i = 0; i < tid; i++) {
            sum += warp_sums[i];
        }
        warp_offsets[tid] = sum;
    }
    __syncthreads();

    // Each thread's full prefix is the sum of its intra-warp scan and the warp's offset
    scalar_t full_prefix = prefix + warp_offsets[warp_id];

    // Compute total sum T of the row using block reduction with __shfl_down_sync()
    // Each thread starts with its own input value
    scalar_t block_val = a;
    for (int offset = 16; offset > 0; offset /= 2) {
        block_val += __shfl_down_sync(0xffffffff, block_val, offset);
    }

    __shared__ scalar_t block_sums[32];
    if (lane == 0) {
        block_sums[warp_id] = block_val;
    }
    __syncthreads();

    // Let the first warp reduce the block_sums to obtain the row total T
    scalar_t total = 0;
    if (tid < num_warps) {
        total = block_sums[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
    }
    __shared__ scalar_t T_shared;
    if (tid == 0) {
        T_shared = total;  // total sum for the row
    }
    __syncthreads();

    // Compute the reverse cumulative sum: out[i] = T - (full_prefix - a)
    if (tid < n) {
        output[base + tid] = T_shared - (full_prefix - a);
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

    // Use the optimized kernel if the cumulative sum is along the last dimension and n <= 1024
    if (dim == ndim - 1 && n <= 1024) {
        int threads = 1;
        while (threads < n) {
            threads *= 2;
        }
        if (threads > 1024) threads = 1024;
        
        dim3 blocks(outer);
        dim3 threadBlock(threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "shfl_reduce_reverse_cumsum_kernel", ([&] {
            shfl_reduce_reverse_cumsum_kernel<scalar_t><<<blocks, threadBlock>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n);
        }));
        return output;
    } else {
        // Fallback to PyTorch's flip-cumsum-flip method for non-ideal cases
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        output = cumsum.flip(dim);
        return output;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with optimized block reduction using __shfl_down_sync (CUDA)");
}
