#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel optimized with variable block sizes for reverse cumulative sum along the last dimension

// Device function for warp-level inclusive scan using shuffle intrinsics
__device__ float warp_inclusive_scan(float val) {
    for (int offset = 1; offset < 32; offset *= 2) {
        float n_val = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x % 32 >= offset) {
            val += n_val;
        }
    }
    return val;
}

// Kernel for performing reverse cumulative sum
__global__ void blocksize_optimized_reverse_cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, const int64_t n) {
    int row = blockIdx.x;
    const int64_t row_offset = row * n;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    __shared__ float warp_sums[32];
    __shared__ float warp_offsets[32];
    __shared__ volatile int next_warp;
    __shared__ float shared_offset;

    if (tid == 0) {
        next_warp = 0;
        shared_offset = 0;
    }
    __syncthreads();

    float val = 0;
    if (tid < n) {
        val = input[row_offset + (n - 1 - tid)];
    }

    val = warp_inclusive_scan(val);

    if (tid < n && (lane == 31 || tid == n - 1)) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (tid < n && lane == 0) {
        while (warp_id != next_warp) { /* busy-wait in shared memory */ }
        warp_offsets[warp_id] = shared_offset;
        atomicAdd((float *)&shared_offset, warp_sums[warp_id]);
        atomicExch((int *)&next_warp, next_warp + 1);
    }
    __syncthreads();

    if (tid < n) {
        float final_val = val + warp_offsets[warp_id];
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

    // Investigate performance with various block sizes (32, 64, 128, 256, 512)
    int optimal_block_size = 256; // example chosen after experimenting on target hardware

    if (dim == ndim - 1 && n <= 1024) {
        int threads = optimal_block_size;

        dim3 blocks(outer);
        dim3 threadBlock(threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "blocksize_optimized_reverse_cumsum_cuda", ([&] {
            blocksize_optimized_reverse_cumsum_kernel<scalar_t><<<blocks, threadBlock>>>(
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
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with optimized block size (CUDA)");
}
