#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that applies manual loop unrolling to speed up the reverse cumulative sum operation
// This kernel performs the reverse operation using warp shuffles and shared memory to compute offsets.

template <typename scalar_t>
__global__ void unrolled_reverse_cumsum_kernel(const scalar_t* __restrict__ input,
                                               scalar_t* __restrict__ output,
                                               int64_t n) {
    int row = blockIdx.x;
    const int64_t row_offset = row * n;

    int tid = threadIdx.x;
    int lane = tid & 31;  // Lane index within the warp

    // Load element in reverse order
    scalar_t val = 0;
    if (tid < n) {
        val = input[row_offset + (n - 1 - tid)];
    }

    // Perform warp-level inclusive scan using shuffle intrinsics with unrolled loop
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t tmp = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += tmp;
        }
    }

    // Each warp's last active thread writes its result to shared memory
    __shared__ scalar_t warp_sums[32];
    int warp_id = tid >> 5;
    if (tid < n && (lane == 31 || tid == n - 1)) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Each thread computes the offset for its warp by summing the totals of previous warps
    if (tid < n) {
        scalar_t warp_offset = 0;
        for (int w = 0; w < warp_id; w++) {
            warp_offset += warp_sums[w];
        }
        scalar_t final_val = val + warp_offset;
        // Write the computed cumulative sum back in the original order
        output[row_offset + (n - 1 - tid)] = final_val;
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

    // Use the optimized kernel only if the cum-sum is performed along the last dimension and n is not large
    if (dim == ndim - 1 && n <= 1024) {
        // Determine number of threads as next power of 2 >= n (capped at 1024)
        int threads = 1;
        while (threads < n) {
            threads *= 2;
        }
        if (threads > 1024) threads = 1024;

        dim3 blocks(outer);
        dim3 threadsPerBlock(threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "unrolled_reverse_cumsum_kernel", ([&] {
            unrolled_reverse_cumsum_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n);
        }));
    } else {
        // Fallback to the flip-cumsum-flip method for non-ideal cases
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        output = cumsum.flip(dim);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with unrolled loops (CUDA)");
}