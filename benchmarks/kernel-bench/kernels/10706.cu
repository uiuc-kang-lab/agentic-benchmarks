#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to compute reverse cumulative sum optimized with different block sizes
// leveraging warp-level shuffles and shared memory for intra-block communication

template <typename scalar_t>
__global__ void optim_blocksize_reverse_cumsum_kernel(const scalar_t* __restrict__ input,
                                                      scalar_t* __restrict__ output,
                                                      int64_t n) {
    // Each block processes one row
    int row = blockIdx.x;
    const int64_t row_offset = row * n;

    int tid = threadIdx.x;
    int lane = tid & 31;      // Lane index within the warp
    int warp_id = tid / 32;   // Warp id

    scalar_t val = 0;
    if (tid < n) {
        val = input[row_offset + (n - 1 - tid)];
    }

    // Perform warp-level inclusive scan using shuffle intrinsics
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t temp = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += temp;
        }
    }

    // Shared memory to store the results of each warp's last thread
    __shared__ scalar_t warp_sums[32];
    if (lane == 31 || tid == n - 1) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Compute the offsets for each warp
    scalar_t warp_offset = 0;
    for (int i = 0; i < warp_id; ++i) {
        warp_offset += warp_sums[i];
    }

    // Add the warp offset to each thread's value
    if (tid < n) {
        scalar_t final_val = val + warp_offset;
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
        int threads_per_block = 256; // Experimenting with a middle-ground block size

        dim3 blocks(outer);
        dim3 threadsPerBlock(threads_per_block);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optim_blocksize_reverse_cumsum_kernel", ([&] {
            optim_blocksize_reverse_cumsum_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
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
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with optimized block size (CUDA)");
}