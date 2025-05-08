#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Using constant memory to store frequent read-only data
__constant__ float constant_data[1024];

// Kernel to compute reverse cumulative sum using constant memory for input data
// This kernel assumes that the dimension size n is <= 1024
// and that the cumsum dimension is contiguous and is the last dimension.

template <typename scalar_t>
__global__ void constant_reverse_cumsum_kernel(const scalar_t* __restrict__ input,
                                               scalar_t* __restrict__ output,
                                               const int64_t n) {
    int row = blockIdx.x;
    const int64_t row_offset = row * n;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    // Load input data into constant memory
    if (tid < n) {
        constant_data[tid] = input[row_offset + (n - 1 - tid)];
    }
    __syncthreads();

    // Load element from constant memory
    scalar_t val = 0;
    if (tid < n) {
        val = constant_data[tid];
    }

    // Perform warp-level inclusive scan with shuffle intrinsics
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t n_val = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += n_val;
        }
    }

    __shared__ scalar_t warp_sums[32];
    __shared__ scalar_t warp_offsets[32];
    __shared__ volatile int next_warp;
    __shared__ scalar_t shared_offset;

    if (tid == 0) {
        next_warp = 0;
        shared_offset = 0;
    }
    __syncthreads();

    if (tid < n && (lane == 31 || tid == n - 1)) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (tid < n && lane == 0) {
        while (warp_id != next_warp) { /* busy-wait */ }
        warp_offsets[warp_id] = shared_offset;
        atomicAdd((scalar_t *)&shared_offset, warp_sums[warp_id]);
        atomicExch((int *)&next_warp, next_warp + 1);
    }
    __syncthreads();

    if (tid < n) {
        scalar_t final_val = val + warp_offsets[warp_id];
        output[row_offset + (n - 1 - tid)] = final_val;
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
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

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "constant_reverse_cumsum_cuda", ([&] {
            constant_reverse_cumsum_kernel<scalar_t><<<blocks, threadBlock>>>(
                x.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), n);
        }));
    } else {
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        output = cumsum.flip(dim);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with constant memory (CUDA)");
}