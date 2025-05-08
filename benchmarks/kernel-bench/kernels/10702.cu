#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for warp-level inclusive scan using shuffle intrinsics
template <typename scalar_t>
__device__ inline scalar_t warp_inclusive_scan(scalar_t val) {
    // Perform an inclusive scan within a warp
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t tmp = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) >= offset) {
            val += tmp;
        }
    }
    return val;
}

// Kernel to compute reverse cumulative sum along the last dimension without using global atomics
// Each block processes one row (from the outer dimension).
// This kernel is designed for contiguous tensors where n (the size along the cumulated dimension) <= 1024.
template <typename scalar_t>
__global__ void warp_no_atomic_reverse_cumsum_kernel(const scalar_t* __restrict__ input,
                                                       scalar_t* __restrict__ output,
                                                       int64_t n) {
    int row = blockIdx.x;
    int64_t row_offset = row * n;

    int tid = threadIdx.x;
    int lane = tid & 31;          // Lane index within the warp
    int warp_id = tid / 32;       // Warp index

    scalar_t val = 0;
    if (tid < n) {
        // Load element in reverse order
        val = input[row_offset + (n - 1 - tid)];
    }

    // Compute warp-level inclusive scan
    scalar_t local_scan = warp_inclusive_scan<scalar_t>(val);

    // Shared memory for storing the last value of each warp and the computed warp offsets
    __shared__ scalar_t warp_sums[32];
    __shared__ scalar_t warp_offsets[32];

    // The last active thread in each warp stores its final scanned value
    if (tid < n && (lane == 31 || tid == n - 1)) {
        warp_sums[warp_id] = local_scan;
    }
    __syncthreads();

    // Thread 0 computes the warp offsets sequentially to avoid any atomic contention
    if (tid == 0) {
        int num_warps = (n + 31) / 32;  
        warp_offsets[0] = 0;
        for (int i = 1; i < num_warps; i++) {
            warp_offsets[i] = warp_offsets[i - 1] + warp_sums[i - 1];
        }
    }
    __syncthreads();

    // Each thread adds the corresponding warp offset and writes the result in the correct order
    if (tid < n) {
        scalar_t final_val = local_scan;
        if (warp_id > 0) {
            final_val += warp_offsets[warp_id];
        }
        output[row_offset + (n - 1 - tid)] = final_val;
    }
}


at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    // Ensure tensor is contiguous and on CUDA
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "dim out of range");

    auto output = at::empty_like(x);

    int64_t n = x.size(dim);
    int64_t outer = x.numel() / n;

    // Use the custom kernel when operating on the last dimension with limited size
    if (dim == x.dim() - 1 && n <= 1024) {
        int threads = 1;
        while (threads < n) {
            threads *= 2;
        }
        if (threads > 1024) threads = 1024;

        dim3 blocks(outer);
        dim3 threadsPerBlock(threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "warp_no_atomic_reverse_cumsum_kernel", ([&] {
            warp_no_atomic_reverse_cumsum_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n);
        }));
    } else {
        // Fallback to the flip-cumsum-flip approach for non-ideal cases
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        output = cumsum.flip(dim);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum using warp scan without global atomics (CUDA)");
}
