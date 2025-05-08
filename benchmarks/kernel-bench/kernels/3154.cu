#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Combined kernel: uses compile-time block size tuning and warp-level reductions
// to efficiently compute the LogSoftmax over the last dimension of the input tensor.

template <typename scalar_t, int BLOCK_SIZE>
__global__ void combined_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    // Each block processes one row (batch element)
    int row = blockIdx.x;
    const scalar_t* input_row = input + row * dim_size;
    scalar_t* output_row = output + row * dim_size;

    // Phase 1: Compute the maximum value using warp-level reduction
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    
    // Each thread processes multiple elements
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        scalar_t val = input_row[idx];
        thread_max = (val > thread_max) ? val : thread_max;
    }

    // Warp-level reduction for maximum using shuffle intrinsics
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(mask, thread_max, offset);
        thread_max = (other > thread_max) ? other : thread_max;
    }

    // Shared memory to gather per-warp maximums
    __shared__ scalar_t warp_max[32];  // Supports up to 32 warps per block
    int warp_id = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    if (lane == 0) {
        warp_max[warp_id] = thread_max;
    }
    __syncthreads();

    // Thread 0 computes the block-wide maximum from warp results
    scalar_t max_val = warp_max[0];
    if (threadIdx.x == 0) {
        int num_warps = (BLOCK_SIZE + warpSize - 1) / warpSize;
        for (int i = 1; i < num_warps; i++) {
            max_val = (warp_max[i] > max_val) ? warp_max[i] : max_val;
        }
        // Store global max in warp_max[0] for broadcast
        warp_max[0] = max_val;
    }
    __syncthreads();
    max_val = warp_max[0];

    // Phase 2: Compute the sum of exponentials (with numerical stability)
    scalar_t thread_sum = 0;
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        thread_sum += exp(input_row[idx] - max_val);
    }

    // Warp-level reduction for sum
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Use shared memory to gather per-warp sums
    __shared__ scalar_t warp_sum[32];
    if (lane == 0) {
        warp_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // Thread 0 sums the warp results to get the total sum
    scalar_t total_sum = 0;
    if (threadIdx.x == 0) {
        int num_warps = (BLOCK_SIZE + warpSize - 1) / warpSize;
        for (int i = 0; i < num_warps; i++) {
            total_sum += warp_sum[i];
        }
        warp_sum[0] = total_sum; // broadcast the total sum
    }
    __syncthreads();
    total_sum = warp_sum[0];
    scalar_t log_sum = log(total_sum);

    // Phase 3: Compute the final LogSoftmax values and write back
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        output_row[idx] = (input_row[idx] - max_val) - log_sum;
    }
}


// Host function: Permutes input tensor so that the specified dimension is last,
// selects an optimal block size based on the dimension size, launches the kernel,
// and then inversely permutes the output to the original layout.

torch::Tensor combined_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
                "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = (dim >= 0) ? dim : dim + ndim;

    // Permute input so that the target dimension is the last dimension
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

    // Select an optimal block size from {32, 64, 128, 256, 512} based on dim_size
    int optimal_block_size = 256; // default value
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
        optimal_block_size = 512; // for larger dims, cap at 512 threads per block
    }

    int blocks = batch_size;
    dim3 grid(blocks);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "combined_logsoftmax_cuda_forward", ([&] {
        if (optimal_block_size == 32) {
            combined_logsoftmax_kernel<scalar_t, 32><<<grid, 32>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 64) {
            combined_logsoftmax_kernel<scalar_t, 64><<<grid, 64>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 128) {
            combined_logsoftmax_kernel<scalar_t, 128><<<grid, 128>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 256) {
            combined_logsoftmax_kernel<scalar_t, 256><<<grid, 256>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 512) {
            combined_logsoftmax_kernel<scalar_t, 512><<<grid, 512>>>(
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
    m.def("forward", &combined_logsoftmax_cuda_forward, "Combined LogSoftmax forward (CUDA)");
}
