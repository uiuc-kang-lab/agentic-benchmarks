#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>


// Warp-level reduction for max (no __syncthreads() needed within a warp)
template <typename scalar_t>
__inline__ __device__ scalar_t warp_reduce_max(scalar_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
template <typename scalar_t>
__inline__ __device__ scalar_t warp_reduce_sum(scalar_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// This kernel computes the LogSoftmax activation with minimized __syncthreads() calls.
// It uses warp-level primitives for intra-warp reductions and employs volatile shared memory with __threadfence_block()
// to broadcast the block-level results without an extra barrier.

template <typename scalar_t, int BLOCK_SIZE>
__global__ void sync_optimized_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    int row = blockIdx.x;
    const scalar_t* in_row = input + row * dim_size;
    scalar_t* out_row = output + row * dim_size;

    // Shared memory to hold per-warp partial results. Its size should be at least (BLOCK_SIZE/warpSize) elements.
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // ----------------------- Phase 1: Max Reduction -----------------------
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = threadIdx.x; i < dim_size; i += BLOCK_SIZE) {
        thread_max = max(thread_max, in_row[i]);
    }

    // Intra-warp reduction (using shuffle, no __syncthreads() needed)
    thread_max = warp_reduce_max(thread_max);

    // Each warp's lane 0 writes its result to shared memory
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        sdata[threadIdx.x >> 5] = thread_max;
    }
    // Ensure all warp leaders have written
    __syncthreads();

    int warp_count = (BLOCK_SIZE + warpSize - 1) / warpSize;
    // Use a volatile pointer to avoid an extra barrier
    volatile scalar_t* vsdata = sdata;
    if (threadIdx.x == 0) {
        scalar_t max_val = vsdata[0];
        for (int i = 1; i < warp_count; i++) {
            max_val = max(max_val, vsdata[i]);
        }
        vsdata[0] = max_val;
        __threadfence_block(); // flush the update to shared memory
    }
    // All threads can now read the final maximum without an extra __syncthreads()
    scalar_t max_val = vsdata[0];

    // ----------------------- Phase 2: Sum Reduction -----------------------
    scalar_t thread_sum = 0;
    for (int i = threadIdx.x; i < dim_size; i += BLOCK_SIZE) {
        thread_sum += exp(in_row[i] - max_val);
    }
    thread_sum = warp_reduce_sum(thread_sum);

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        sdata[threadIdx.x >> 5] = thread_sum;
    }
    __syncthreads();

    int warp_count2 = (BLOCK_SIZE + warpSize - 1) / warpSize;
    volatile scalar_t* vsdata2 = sdata;
    if (threadIdx.x == 0) {
        scalar_t sum_val = vsdata2[0];
        for (int i = 1; i < warp_count2; i++) {
            sum_val += vsdata2[i];
        }
        vsdata2[0] = sum_val;
        __threadfence_block();
    }
    scalar_t sum_val = vsdata2[0];
    scalar_t log_sum = log(sum_val);

    // ----------------------- Phase 3: Final LogSoftmax Output -----------------------
    for (int i = threadIdx.x; i < dim_size; i += BLOCK_SIZE) {
        out_row[i] = (in_row[i] - max_val) - log_sum;
    }
}

// Host function: Permute the input so that the reduction dimension is last, launch the kernel, and inverse permute.
// The kernel uses dynamic shared memory size equal to ((BLOCK_SIZE / warpSize) * sizeof(scalar_t)).

torch::Tensor optimized_sync_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = dim >= 0 ? dim : dim + ndim;

    // Permute input so that the target dimension is last
    std::vector<int64_t> permute_dims;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            permute_dims.push_back(i);
        }
    }
    permute_dims.push_back(dim);

    input = input.permute(permute_dims).contiguous();
    int64_t batch_size = input.numel() / input.size(-1);
    int64_t dim_size = input.size(-1);

    auto output = torch::empty_like(input);

    // Choose block size (must be a multiple of warpSize) based on dim_size.
    int optimal_block_size = 256;
    if (dim_size <= 32) {
        optimal_block_size = 32;
    } else if (dim_size <= 64) {
        optimal_block_size = 64;
    } else if (dim_size <= 128) {
        optimal_block_size = 128;
    } else if (dim_size <= 256) {
        optimal_block_size = 256;
    } else if (dim_size <= 512) {
        optimal_block_size = 512;
    } else {
        optimal_block_size = 512;
    }

    const int blocks = batch_size;
    size_t shared_mem_size = ((optimal_block_size + warpSize - 1) / warpSize) * sizeof(float);
    // Use the appropriate shared memory size based on the data type (float/double).
    if (input.scalar_type() == torch::kFloat64) {
        shared_mem_size = ((optimal_block_size + warpSize - 1) / warpSize) * sizeof(double);
    }

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_sync_logsoftmax_cuda_forward", ([&] {
        if (optimal_block_size == 32) {
            sync_optimized_logsoftmax_kernel<scalar_t, 32><<<blocks, 32, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 64) {
            sync_optimized_logsoftmax_kernel<scalar_t, 64><<<blocks, 64, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 128) {
            sync_optimized_logsoftmax_kernel<scalar_t, 128><<<blocks, 128, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 256) {
            sync_optimized_logsoftmax_kernel<scalar_t, 256><<<blocks, 256, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 512) {
            sync_optimized_logsoftmax_kernel<scalar_t, 512><<<blocks, 512, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        }
    }));

    // Inverse permute to restore the original layout
    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_sync_logsoftmax_cuda_forward, "Optimized Sync LogSoftmax forward (CUDA)");
}
