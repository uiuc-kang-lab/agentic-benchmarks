#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// This kernel efficiently computes the LogSoftmax over the last dimension
// using warp-level reductions and shared memory to compute both the maximum
// and the sum of exponentials in a numerically stable way. It combines the best
// ideas from the two provided kernels.

template <typename scalar_t, int BLOCK_SIZE>
__global__ void efficient_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    // Each block processes one row (batch element)
    int row = blockIdx.x;
    const scalar_t* input_row = input + row * dim_size;
    scalar_t* output_row = output + row * dim_size;

    // Shared memory arrays for per-warp reductions (max and sum)
    __shared__ scalar_t shared_max[32];  // supports up to 32 warps per block
    __shared__ scalar_t shared_sum[32];

    // Phase 1: Compute maximum value in the row using warp-level reduction
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        thread_max = max(thread_max, input_row[idx]);
    }

    // Perform warp-level reduction for max using shuffle intrinsics
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(mask, thread_max, offset);
        thread_max = max(thread_max, other);
    }

    // Each warp writes its maximum value to shared memory
    int warp_id = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    if (lane == 0) {
        shared_max[warp_id] = thread_max;
    }
    __syncthreads();

    // First few threads (one per warp) reduce the per-warp maximums
    if (threadIdx.x < (BLOCK_SIZE + warpSize - 1) / warpSize) {
        thread_max = shared_max[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            scalar_t other = __shfl_down_sync(mask, thread_max, offset);
            thread_max = max(thread_max, other);
        }
        if (threadIdx.x == 0) {
            shared_max[0] = thread_max;  // store final block maximum
        }
    }
    __syncthreads();

    scalar_t max_val = shared_max[0];

    // Phase 2: Compute the sum of exp(x - max_val) using warp-level reduction
    scalar_t thread_sum = 0;
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        thread_sum += exp(input_row[idx] - max_val);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Each warp writes its partial sum to shared memory
    if (lane == 0) {
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // Reduce the warp sums into a total sum
    if (threadIdx.x < (BLOCK_SIZE + warpSize - 1) / warpSize) {
        thread_sum = shared_sum[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(mask, thread_sum, offset);
        }
        if (threadIdx.x == 0) {
            shared_sum[0] = thread_sum;  // total sum across the row
        }
    }
    __syncthreads();

    scalar_t total_sum = shared_sum[0];
    scalar_t log_sum = log(total_sum);

    // Phase 3: Write the final LogSoftmax values
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        output_row[idx] = (input_row[idx] - max_val) - log_sum;
    }
}

// Host function: Permutes the input tensor so that the target dimension is the last,
// selects an optimal block size, launches the kernel, then inversely permutes the output.

torch::Tensor efficient_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = (dim >= 0) ? dim : dim + ndim;

    // Permute so that the selected dimension is last for coalesced memory access
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

    // Select an optimal block size based on dim_size
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
        optimal_block_size = 512; // cap at 512 threads per block
    }

    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "efficient_logsoftmax_cuda_forward", ([&] {
        if (optimal_block_size == 32) {
            efficient_logsoftmax_kernel<scalar_t, 32><<<blocks, 32>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 64) {
            efficient_logsoftmax_kernel<scalar_t, 64><<<blocks, 64>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 128) {
            efficient_logsoftmax_kernel<scalar_t, 128><<<blocks, 128>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 256) {
            efficient_logsoftmax_kernel<scalar_t, 256><<<blocks, 256>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 512) {
            efficient_logsoftmax_kernel<scalar_t, 512><<<blocks, 512>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        }
    }));

    // Inverse permutation to restore original tensor layout
    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &efficient_logsoftmax_cuda_forward, "Efficient LogSoftmax forward (CUDA)");
}
