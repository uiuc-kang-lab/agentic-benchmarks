#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// This kernel leverages shared memory to cache frequently reused row data
// and reduce global memory latency. It has two branches: one for when the
// entire row fits in shared memory, and a tiling branch for larger rows.

template <typename scalar_t>
__global__ void smem_optimized_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    // Each block processes one row (batch element).
    int batch_idx = blockIdx.x;
    const int row_offset = batch_idx * dim_size;

    // Dynamically allocated shared memory. It will be used either to cache an entire
    // row (if it fits) or as a reduction buffer in the tiling path.
    extern __shared__ char shared[];
    scalar_t* tile = reinterpret_cast<scalar_t*>(shared);

    // We use blockDim.x as the number of threads in the block.
    // Branch 1: Entire row fits into shared memory
    if (dim_size <= blockDim.x) {
        // Load the entire row into shared memory
        for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
            tile[i] = input[row_offset + i];
        }
        __syncthreads();

        // Compute the maximum value in the row using tree reduction
        // We'll perform the reduction in shared memory.
        int n = dim_size;
        for (int stride = 1; stride < n; stride *= 2) {
            int index = threadIdx.x * (stride * 2);
            if (index < n && (index + stride) < n) {
                tile[index] = max(tile[index], tile[index + stride]);
            }
            __syncthreads();
        }
        scalar_t max_val = tile[0];
        __syncthreads();

        // Compute the sum of exponentials in shared memory
        // Reload the row from global memory if needed, or recompute using the input
        for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
            // Compute exp(x - max) and store back into shared memory
            tile[i] = exp(input[row_offset + i] - max_val);
        }
        __syncthreads();

        // Reduce to compute the sum
        n = dim_size;
        for (int stride = 1; stride < n; stride *= 2) {
            int index = threadIdx.x * (stride * 2);
            if (index < n && (index + stride) < n) {
                tile[index] += tile[index + stride];
            }
            __syncthreads();
        }
        scalar_t sum_val = tile[0];
        __syncthreads();

        scalar_t log_sum = log(sum_val);

        // Write final log-softmax values to output from global memory
        for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
            output[row_offset + i] = (input[row_offset + i] - max_val) - log_sum;
        }
    } else {
        // Branch 2: Tiling approach for rows that don't fit entirely in shared memory
        // Phase 1: Compute maximum over the row
        scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
        for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
            local_max = max(local_max, input[row_offset + i]);
        }
        // Store each thread's partial max in shared memory
        tile[threadIdx.x] = local_max;
        __syncthreads();
        
        // Reduction in shared memory to compute the row maximum
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                tile[threadIdx.x] = max(tile[threadIdx.x], tile[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        scalar_t max_val = tile[0];
        __syncthreads();

        // Phase 2: Compute sum of exponentials of (x - max) using tiling
        scalar_t local_sum = 0;
        for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
            local_sum += exp(input[row_offset + i] - max_val);
        }
        tile[threadIdx.x] = local_sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                tile[threadIdx.x] += tile[threadIdx.x + stride];
            }
            __syncthreads();
        }
        scalar_t sum_val = tile[0];
        __syncthreads();

        scalar_t log_sum = log(sum_val);

        // Phase 3: Write the final log-softmax output
        for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
            output[row_offset + i] = (input[row_offset + i] - max_val) - log_sum;
        }
    }
}

// Host function: Permutes input so that the reduction dimension is last, determines optimal block configuration,
// leverages shared memory allocation to cache input rows when possible, and launches the CUDA kernel.

torch::Tensor smem_optimized_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = (dim >= 0) ? dim : dim + ndim;

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

    // Choose an optimal number of threads per block.
    // If the row fits in shared memory, we set blockDim.x to dim_size; otherwise, we cap it at 512.
    int threads = (dim_size <= 512) ? static_cast<int>(dim_size) : 512;
    dim3 blocks(batch_size);
    dim3 threadsPerBlock(threads);

    // Determine the size of shared memory needed:
    // For the caching branch, we need (dim_size * sizeof(scalar_t)).
    // For the tiling branch, we need (threads * sizeof(scalar_t)).
    size_t smem_size = (dim_size <= threads) ? (dim_size * sizeof(float)) : (threads * sizeof(float));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "smem_optimized_logsoftmax_cuda_forward", ([&] {
        smem_optimized_logsoftmax_kernel<scalar_t><<<blocks, threadsPerBlock, smem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size);
    }));

    // Inverse permute to restore the original tensor layout
    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smem_optimized_logsoftmax_cuda_forward, "Shared Memory Optimized LogSoftmax forward (CUDA)");
}
