#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// This kernel computes LogSoftmax with minimal use of atomic operations. It uses warp-level intrinsics
// and shared memory to perform reductions for both the maximum value and the sum of exponentials.
// Atomic operations are avoided by performing a two-step reduction: first within each warp using __shfl_down_sync,
// and then across warps using shared memory. This minimizes global memory contention and synchronization overhead.


template <typename scalar_t, int BLOCK_SIZE>
__global__ void atomic_free_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    // Each block handles one row (batch element)
    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Phase 1: Compute the maximum value for numerical stability
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        thread_max = max(thread_max, input_row[idx]);
    }

    // Warp-level reduction for maximum using shuffle intrinsics
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(mask, thread_max, offset);
        thread_max = max(thread_max, other);
    }

    // Each warp writes its result to shared memory
    __shared__ scalar_t shared_max[BLOCK_SIZE / 32];
    int warp_id = threadIdx.x / warpSize;
    if ((threadIdx.x % warpSize) == 0) {
        shared_max[warp_id] = thread_max;
    }
    __syncthreads();

    // First warp reduces the per-warp maximums
    scalar_t max_val;
    if (threadIdx.x < (BLOCK_SIZE / 32)) {
        max_val = shared_max[threadIdx.x];
        if (threadIdx.x == 0) {
            for (int i = 1; i < (BLOCK_SIZE / 32); i++) {
                max_val = max(max_val, shared_max[i]);
            }
            shared_max[0] = max_val;
        }
    }
    __syncthreads();
    max_val = shared_max[0];

    // Phase 2: Compute the sum of exp(x - max_val) using warp-level reduction
    scalar_t thread_sum = 0;
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        thread_sum += exp(input_row[idx] - max_val);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    __shared__ scalar_t shared_sum[BLOCK_SIZE / 32];
    if ((threadIdx.x % warpSize) == 0) {
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp reduces the per-warp sums
    scalar_t sum_val;
    if (threadIdx.x < (BLOCK_SIZE / 32)) {
        sum_val = shared_sum[threadIdx.x];
        if (threadIdx.x == 0) {
            for (int i = 1; i < (BLOCK_SIZE / 32); i++) {
                sum_val += shared_sum[i];
            }
            shared_sum[0] = sum_val;
        }
    }
    __syncthreads();
    sum_val = shared_sum[0];
    scalar_t log_sum = log(sum_val);

    // Phase 3: Compute the final LogSoftmax values
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        output_row[idx] = (input_row[idx] - max_val) - log_sum;
    }
}


// Host function: permute dimensions, launch the kernel, and restore original layout

torch::Tensor atomic_free_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = dim >= 0 ? dim : dim + ndim;

    // Permute input so that the target dimension is the last
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

    // Choose an optimal block size from {32, 64, 128, 256, 512}
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

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "atomic_free_logsoftmax_cuda_forward", ([&] {
        if (optimal_block_size == 32) {
            atomic_free_logsoftmax_kernel<scalar_t, 32><<<blocks, 32>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 64) {
            atomic_free_logsoftmax_kernel<scalar_t, 64><<<blocks, 64>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 128) {
            atomic_free_logsoftmax_kernel<scalar_t, 128><<<blocks, 128>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 256) {
            atomic_free_logsoftmax_kernel<scalar_t, 256><<<blocks, 256>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 512) {
            atomic_free_logsoftmax_kernel<scalar_t, 512><<<blocks, 512>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        }
    }));

    // Inverse permutation to restore the original tensor layout
    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &atomic_free_logsoftmax_cuda_forward, "Atomic Free LogSoftmax forward (CUDA)");
}
