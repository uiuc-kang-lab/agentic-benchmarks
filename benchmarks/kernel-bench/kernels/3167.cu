#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Kernel optimized using warp-level primitives and shared memory for efficient reduction

template <typename scalar_t, int BLOCK_SIZE>
__global__ void warp_optimized_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    __shared__ scalar_t shared_max;
    __shared__ scalar_t shared_sum;
    
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    scalar_t local_sum = 0;
    
    // Phase 1: Compute maximum value using warp-level reduction
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        scalar_t val = input_row[idx];
        local_max = max(local_max, val);
    }

    // Perform warp-level reduction to find the maximum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }
    
    // Write the warp-level max to shared memory
    if (threadIdx.x % warpSize == 0) {
        atomicMax(&shared_max, local_max);
    }
    __syncthreads();
    max_val = shared_max;

    // Phase 2: Compute sum of exponentials using warp-level reduction
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        scalar_t exp_val = exp(input_row[idx] - max_val);
        local_sum += exp_val;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&shared_sum, local_sum);
    }
    __syncthreads();
    scalar_t sum_val = shared_sum;
    scalar_t log_sum = log(sum_val);

    // Phase 3: Compute final LogSoftmax values
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        output_row[idx] = (input_row[idx] - max_val) - log_sum;
    }
}

// Host function

torch::Tensor warp_optimized_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
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

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "warp_optimized_logsoftmax_cuda_forward", ([&] {
        if (optimal_block_size == 32) {
            warp_optimized_logsoftmax_kernel<scalar_t, 32><<<blocks, 32>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 64) {
            warp_optimized_logsoftmax_kernel<scalar_t, 64><<<blocks, 64>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 128) {
            warp_optimized_logsoftmax_kernel<scalar_t, 128><<<blocks, 128>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 256) {
            warp_optimized_logsoftmax_kernel<scalar_t, 256><<<blocks, 256>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 512) {
            warp_optimized_logsoftmax_kernel<scalar_t, 512><<<blocks, 512>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        }
    }));

    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_optimized_logsoftmax_cuda_forward, "Warp Optimized LogSoftmax forward (CUDA)");
}
