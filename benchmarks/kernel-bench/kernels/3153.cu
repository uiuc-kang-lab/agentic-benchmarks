#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// This kernel uses 2D thread block indexing with warp-level intrinsics for efficient reduction
// Each block processes one row (batch element) and uses a 2D block with blockDim.x = 32 (warp size)
// and blockDim.y = number of warps (computed based on dim_size, capped to 32).

// The kernel computes the maximum value and sum of exponentials (for numerical stability) using
// warp shuffle reductions and a final per-block reduction, then writes the log softmax result.

template <typename scalar_t>
__global__ void grid2d_log_softmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    // Each block corresponds to one batch element (one row)
    int batch_idx = blockIdx.x;
    const scalar_t* row = input + batch_idx * dim_size;
    scalar_t* out_row = output + batch_idx * dim_size;

    // 2D block: flatten thread index
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;

    // Step 1: Compute maximum value in the row
    // Each thread processes multiple elements using a grid-stride loop
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int j = tid; j < dim_size; j += total_threads) {
        local_max = (row[j] > local_max) ? row[j] : local_max;
    }

    // Use warp shuffle reduction to reduce local max within each warp
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(mask, local_max, offset);
        local_max = (local_max > other) ? local_max : other;
    }

    // Allocate shared memory for storing per-warp max results
    // We'll allocate 2 * blockDim.y elements: first blockDim.y for max, next for sum
    extern __shared__ char smem[];
    scalar_t* smax = reinterpret_cast<scalar_t*>(smem);  // size: blockDim.y elements
    int warp_id = threadIdx.y;  // each warp has 1 result (lane 0 writes)
    if (threadIdx.x == 0) {
        smax[warp_id] = local_max;
    }
    __syncthreads();

    // Final reduction of max performed by a single thread (e.g., thread (0,0))
    __shared__ scalar_t max_val;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        scalar_t tmp_max = smax[0];
        for (int i = 1; i < blockDim.y; i++) {
            tmp_max = (smax[i] > tmp_max) ? smax[i] : tmp_max;
        }
        max_val = tmp_max;
    }
    __syncthreads();

    // Step 2: Compute the sum of exp(x - max_val) for numerical stability
    scalar_t local_sum = 0;
    for (int j = tid; j < dim_size; j += total_threads) {
        local_sum += exp(row[j] - max_val);
    }
    // Warp-level reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Use second part of shared memory for storing per-warp sums
    scalar_t* ssum = smax + blockDim.y;  // ssum starts right after smax
    if (threadIdx.x == 0) {
        ssum[warp_id] = local_sum;
    }
    __syncthreads();

    __shared__ scalar_t sum_val;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        scalar_t tmp_sum = 0;
        for (int i = 0; i < blockDim.y; i++) {
            tmp_sum += ssum[i];
        }
        sum_val = tmp_sum;
    }
    __syncthreads();

    scalar_t log_sum = log(sum_val);

    // Step 3: Compute final log softmax output
    for (int j = tid; j < dim_size; j += total_threads) {
        out_row[j] = (row[j] - max_val) - log_sum;
    }
}

// Host function to launch the kernel
// This function permutes the input so that the reduction dimension is last, computes the optimal 2D block
// dimensions based on the dimension size, and then launches the kernel with appropriate shared memory.

torch::Tensor grid2d_log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = dim >= 0 ? dim : dim + ndim;

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

    // Determine 2D block dimensions
    // Use blockDim.x = 32 (warp size) and blockDim.y = number of warps = ceil(dim_size/32), capped at 32
    int warp_dim = 32;
    int num_warps = (dim_size + warp_dim - 1) / warp_dim;
    if (num_warps > 32) num_warps = 32;
    dim3 threads(warp_dim, num_warps);
    dim3 blocks(batch_size);

    // Shared memory: 2 arrays of size (num_warps) for max and sum reductions
    size_t shared_mem_size = 2 * num_warps * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid2d_log_softmax_forward_cuda", ([&] {
        size_t smem_size = 2 * num_warps * sizeof(scalar_t);
        grid2d_log_softmax_forward_kernel<scalar_t><<<blocks, threads, smem_size>>>(
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
    m.def("forward", &grid2d_log_softmax_cuda_forward, "LogSoftmax forward (CUDA) with 2D grid indexing");
}
