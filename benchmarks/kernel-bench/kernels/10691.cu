#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes the reverse cumulative sum for each row (assumed along the last dimension).
// It loads elements in reverse order, performs an intra-warp inclusive scan using warp-level primitives,
// computes each warp's total using __shfl_down_sync, and then uses shared memory to aggregate
// warp totals as offsets. The final result is written back in the original order.

template <typename scalar_t>
__global__ void optimized_reverse_cumsum_kernel(const scalar_t* __restrict__ input,
                                                  scalar_t* __restrict__ output,
                                                  const int64_t n) {
    // Allocate shared memory for warp-level totals
    extern __shared__ char shared[];
    scalar_t* warp_partial = reinterpret_cast<scalar_t*>(shared);

    int row = blockIdx.x;               // Each block processes one row
    int tid = threadIdx.x;
    int lane = tid & 31;                // Lane index within the warp
    int warpId = tid >> 5;              // Warp index

    // Load element in reversed order (if tid < n), else zero.
    scalar_t val = (tid < n) ? input[row * n + (n - 1 - tid)] : scalar_t(0);

    // Intra-warp inclusive scan using __shfl_up_sync.
    // Each thread adds the value from a lower lane by a power-of-two offset.
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t temp = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += temp;
        }
    }

    // Compute the total sum of this warp using __shfl_down_sync reduction.
    scalar_t warp_sum = val;
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }

    // The first thread in each warp writes its warp total into shared memory.
    if (lane == 0) {
        warp_partial[warpId] = warp_sum;
    }
    __syncthreads();

    // Compute the prefix sum of warp totals; this gives the offset for each warp.
    int numWarps = (blockDim.x + 31) / 32;
    if (tid == 0) {
        scalar_t acc = 0;
        for (int i = 0; i < numWarps; i++) {
            scalar_t temp = warp_partial[i];
            warp_partial[i] = acc;
            acc += temp;
        }
    }
    __syncthreads();

    // Each thread retrieves the offset for its warp and adds it to its local scan result.
    scalar_t warp_offset = warp_partial[warpId];
    val += warp_offset;

    // Write the result back in the original order (flipping the index back).
    if (tid < n) {
        output[row * n + (n - 1 - tid)] = val;
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    // This optimized kernel supports only the last-dimension reverse cumsum.
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    const int ndim = x.dim();
    TORCH_CHECK(dim == ndim - 1, "Optimized kernel supports only reverse cumsum along the last dimension");

    auto output = at::empty_like(x);
    int64_t n = x.size(dim);
    int64_t outer = x.numel() / n;  // Number of rows

    // Determine block size: choose next power of 2 >= n, capped at 1024.
    int threads = 1;
    while (threads < n) {
        threads *= 2;
    }
    if (threads > 1024) threads = 1024;

    int numWarps = (threads + 31) / 32;
    size_t sharedMemSize = numWarps * sizeof(float); // Allocate shared memory per block

    dim3 blocks(outer);
    dim3 threadBlock(threads);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_reverse_cumsum_kernel", ([&] {
        optimized_reverse_cumsum_kernel<scalar_t><<<blocks, threadBlock, numWarps * sizeof(scalar_t)>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            n);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Optimized reverse cumulative sum using shared memory and warp-level primitives (CUDA)");
}
