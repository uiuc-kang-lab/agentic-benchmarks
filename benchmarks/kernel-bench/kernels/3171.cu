#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Optimized combined kernel using shared memory, warp-level reductions and dynamic block size

template <typename scalar_t>
__global__ void optimized_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size,
    int effective_block_size) {

    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    extern __shared__ scalar_t shared_memory[];  // Use dynamically allocated shared memory
    scalar_t* warp_max = shared_memory;
    scalar_t* warp_sum = shared_memory + (effective_block_size / warpSize);

    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();

    for (int idx = threadIdx.x; idx < dim_size; idx += effective_block_size) {
        local_max = fmaxf(local_max, input_row[idx]);
    }

    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(mask, local_max, offset);
        local_max = fmaxf(local_max, other);
    }

    int warp_id = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();

    if (threadIdx.x == 0) {
        scalar_t block_max = -std::numeric_limits<scalar_t>::infinity();
        int num_warps = (effective_block_size + warpSize - 1) / warpSize;
        for (int i = 0; i < num_warps; i++) {
            block_max = fmaxf(block_max, warp_max[i]);
        }
        warp_max[0] = block_max; // Use warp_max[0] to store the block-wide maximum
    }
    __syncthreads();
    scalar_t max_val = warp_max[0];

    scalar_t local_sum = 0;
    for (int idx = threadIdx.x; idx < dim_size; idx += effective_block_size) {
        local_sum += expf(input_row[idx] - max_val);
    }

    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    if (lane == 0) warp_sum[warp_id] = local_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        scalar_t block_sum = 0;
        int num_warps = (effective_block_size + warpSize - 1) / warpSize;
        for (int i = 0; i < num_warps; i++) {
            block_sum += warp_sum[i];
        }
        warp_sum[0] = block_sum; // Broadcast the block-wide sum
    }
    __syncthreads();
    scalar_t sum = warp_sum[0];
    scalar_t log_sum = logf(sum);

    for (int idx = threadIdx.x; idx < dim_size; idx += effective_block_size) {
        output_row[idx] = input_row[idx] - max_val - log_sum;
    }
}

// Host function

torch::Tensor optimized_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

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

    int optimal_block_size = 1;
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
        optimal_block_size = 1024; // Increase max threads for very large dims
    }

    dim3 blocks(batch_size);
    int smem_size = optimal_block_size / warpSize * sizeof(scalar_t) * 2;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_logsoftmax_cuda_forward", ([&] {
        optimized_logsoftmax_kernel<scalar_t><<<blocks, optimal_block_size, smem_size, stream>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            optimal_block_size);
    }));

    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}