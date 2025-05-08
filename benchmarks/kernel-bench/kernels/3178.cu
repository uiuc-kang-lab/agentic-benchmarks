#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// This kernel uses shared memory for intra-block reductions and warp-level primitives
// for the final stages, minimizing global atomic operations and reducing contention.

template <typename scalar_t, int BLOCK_SIZE>
__global__ void shared_warp_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Shared memory for reduction
    __shared__ scalar_t smax[BLOCK_SIZE / 32];
    __shared__ scalar_t ssum[BLOCK_SIZE / 32];

    // Each thread finds the local maximum
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        local_max = max(local_max, input_row[idx]);
    }

    // Warp-level reduction within each warp
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(mask, local_max, offset));
    }

    // Store the local maxima from each warp
    int warp_id = threadIdx.x / warpSize;
    if (threadIdx.x % warpSize == 0) {
        smax[warp_id] = local_max;
    }
    __syncthreads();

    // Use first warp to reduce the maxima from each warp
    scalar_t block_max = (threadIdx.x < BLOCK_SIZE / warpSize) ? smax[threadIdx.x] : -std::numeric_limits<scalar_t>::infinity();
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        block_max = max(block_max, __shfl_down_sync(mask, block_max, offset));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        smax[0] = block_max;
    }
    __syncthreads();
    block_max = smax[0];

    // Compute exponential sums
    scalar_t local_sum = 0;
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        local_sum += exp(input_row[idx] - block_max);
    }

    // Warp-level reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    if (threadIdx.x % warpSize == 0) {
        ssum[warp_id] = local_sum;
    }
    __syncthreads();

    // Reduce the sums from each warp
    scalar_t block_sum = (threadIdx.x < BLOCK_SIZE / warpSize) ? ssum[threadIdx.x] : 0;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        block_sum += __shfl_down_sync(mask, block_sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        ssum[0] = block_sum;
    }
    __syncthreads();
    block_sum = ssum[0];
    scalar_t log_sum = log(block_sum);

    // Write back the final LogSoftmax values
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        output_row[idx] = (input_row[idx] - block_max) - log_sum;
    }
}

// Host function to handle the kernel launch and tensor permutation

torch::Tensor shared_warp_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
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

    // Choose optimal block size considering the warp efficiency
    int optimal_block_size = 256; // default
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

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "shared_warp_logsoftmax_cuda_forward", ([&] {
        if (optimal_block_size == 32) {
            shared_warp_logsoftmax_kernel<scalar_t, 32><<<blocks, 32>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 64) {
            shared_warp_logsoftmax_kernel<scalar_t, 64><<<blocks, 64>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 128) {
            shared_warp_logsoftmax_kernel<scalar_t, 128><<<blocks, 128>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 256) {
            shared_warp_logsoftmax_kernel<scalar_t, 256><<<blocks, 256>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 512) {
            shared_warp_logsoftmax_kernel<scalar_t, 512><<<blocks, 512>>>(
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
    m.def("forward", &shared_warp_logsoftmax_cuda_forward, "Shared Warp LogSoftmax forward (CUDA)");
}