#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// This kernel avoids the use of global atomic operations by performing warp-level reduction
// and a subsequent block-level reduction in shared memory. Each warp computes its partial
// maximum and sum, which are then combined by a single thread without resorting to atomics.

template <typename scalar_t, int BLOCK_SIZE>
__global__ void min_atomic_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Shared memory arrays for storing per-warp partial results
    __shared__ scalar_t warp_max[BLOCK_SIZE / 32];
    __shared__ scalar_t warp_sum[BLOCK_SIZE / 32];

    // Phase 1: Compute maximum value in the row for numerical stability
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        local_max = max(local_max, input_row[idx]);
    }

    unsigned int mask = 0xffffffff;
    // Warp-level reduction for maximum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(mask, local_max, offset));
    }

    int warp_id = threadIdx.x / warpSize;
    if ((threadIdx.x & (warpSize - 1)) == 0) {  // one thread per warp
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    // Thread 0 reduces the per-warp maximums
    if (threadIdx.x == 0) {
        scalar_t block_max = -std::numeric_limits<scalar_t>::infinity();
        int num_warps = BLOCK_SIZE / warpSize;
        for (int i = 0; i < num_warps; ++i) {
            block_max = max(block_max, warp_max[i]);
        }
        warp_max[0] = block_max;  // store final maximum
    }
    __syncthreads();
    scalar_t final_max = warp_max[0];

    // Phase 2: Compute the sum of exp(x - max) using similar warp-level reduction
    scalar_t local_sum = 0;
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        local_sum += exp(input_row[idx] - final_max);
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        scalar_t total_sum = 0;
        int num_warps = BLOCK_SIZE / warpSize;
        for (int i = 0; i < num_warps; ++i) {
            total_sum += warp_sum[i];
        }
        warp_sum[0] = total_sum;
    }
    __syncthreads();
    scalar_t final_sum = warp_sum[0];
    scalar_t log_sum = log(final_sum);

    // Phase 3: Compute final LogSoftmax values and write back
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        output_row[idx] = (input_row[idx] - final_max) - log_sum;
    }
}

// Host function to set up kernel launch. It permutes the input so that the target dimension is last,
// launches the kernel with an optimal block size, and then inversely permutes the output to restore the
// original tensor layout.

torch::Tensor min_atomic_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64"
    );

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = dim >= 0 ? dim : dim + ndim;

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

    // Choose an optimal block size based on the dimension size
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

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "min_atomic_logsoftmax_cuda_forward", ([&] {
        if (optimal_block_size == 32) {
            min_atomic_logsoftmax_kernel<scalar_t, 32><<<blocks, 32>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 64) {
            min_atomic_logsoftmax_kernel<scalar_t, 64><<<blocks, 64>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 128) {
            min_atomic_logsoftmax_kernel<scalar_t, 128><<<blocks, 128>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 256) {
            min_atomic_logsoftmax_kernel<scalar_t, 256><<<blocks, 256>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 512) {
            min_atomic_logsoftmax_kernel<scalar_t, 512><<<blocks, 512>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        }
    }));

    // Restore the original tensor layout via inverse permutation
    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &min_atomic_logsoftmax_cuda_forward, "Min Atomic LogSoftmax forward (CUDA)");
}
