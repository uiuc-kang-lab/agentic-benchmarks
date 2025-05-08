#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes the reverse cumulative sum along the last dimension for each row using
// warp-level scan and uses atomic operations on shared memory to combine the per-warp results.
// The atomic operations here are used only in shared memory to order warp results, avoiding global atomics.

// Note: This kernel assumes that the cumsum dimension is contiguous and that n (the size along that dimension) 
// is not larger than the block size (we cap it to <= 1024). For larger or non-contiguous cases, fallback to
// the flip+cumsum+flip approach.

template <typename scalar_t>
__global__ void atomic_reverse_cumsum_kernel(const scalar_t* __restrict__ input,
                                               scalar_t* __restrict__ output,
                                               const int64_t n) {
    // Each block processes one row
    int row = blockIdx.x;
    const int64_t row_offset = row * n;
    int tid = threadIdx.x;
    int lane = tid & 31;       // lane index within a warp
    int warp_id = tid >> 5;    // warp index within the block

    // Allocate shared memory for per-warp sums and offsets
    __shared__ scalar_t warp_sums[32];
    __shared__ scalar_t warp_offsets[32];
    // Shared variables for atomic ordering across warp leaders (in shared memory only)
    // Using volatile to ensure the spin loop sees updated values
    __shared__ volatile int next_warp;  
    __shared__ scalar_t shared_offset;  

    if (tid == 0) {
        next_warp = 0;
        shared_offset = 0;
    }
    __syncthreads();

    // Load element in reverse order if within bounds
    scalar_t val = 0;
    if (tid < n) {
        // Reverse index: thread tid loads element from index (n - 1 - tid)
        val = input[row_offset + (n - 1 - tid)];
    }

    // Perform warp-level inclusive scan with shuffle intrinsics
    // This computes the prefix sum within each warp for the reversed order
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t n_val = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += n_val;
        }
    }

    // The last active thread in each warp records the warp's total sum
    if (tid < n && (lane == 31 || tid == n - 1)) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Each warp leader (lane 0) computes its warp offset using atomic operations in shared memory
    // to ensure that the offsets are accumulated in increasing warp_id order.
    if (tid < n && lane == 0) {
        // Spin until it is this warp's turn
        while (warp_id != next_warp) { /* busy-wait in shared memory */ }
        // The offset for this warp is the current accumulated shared_offset
        warp_offsets[warp_id] = shared_offset;
        // Atomically add this warp's total to the shared_offset
        // Since we're in shared memory and the number of warps is small (max 32), contention is minimal
        atomicAdd((scalar_t *)&shared_offset, warp_sums[warp_id]);
        // Signal that the next warp can proceed
        atomicExch((int *)&next_warp, next_warp + 1);
    }
    __syncthreads();

    // Each thread adds its warp's offset to its scanned value
    if (tid < n) {
        scalar_t final_val = val + warp_offsets[warp_id];
        // Write back in reverse order to produce the correct reverse cumulative sum
        output[row_offset + (n - 1 - tid)] = final_val;
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    // Ensure tensor is contiguous and on CUDA
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");

    const int ndim = x.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    auto output = at::empty_like(x);

    int64_t n = x.size(dim);
    int64_t outer = x.numel() / n;

    // Fast path: if cumsum is along the last dimension and n <= 1024, use the optimized kernel
    if (dim == ndim - 1 && n <= 1024) {
        // Determine the number of threads as the next power of 2 >= n (capped at 1024)
        int threads = 1;
        while (threads < n) {
            threads *= 2;
        }
        if (threads > 1024) threads = 1024;

        // Each block processes one row
        dim3 blocks(outer);
        dim3 threadBlock(threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "atomic_reverse_cumsum_cuda", ([&] {
            atomic_reverse_cumsum_kernel<scalar_t><<<blocks, threadBlock>>>(
                x.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), n);
        }));
    } else {
        // Fallback to PyTorch's built-in operations for non-contiguous or larger cases
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        output = cumsum.flip(dim);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with atomic ops in shared memory (CUDA)");
}
