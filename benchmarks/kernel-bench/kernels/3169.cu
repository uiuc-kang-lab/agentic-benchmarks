#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// This kernel uses shared memory for intra-block reductions and warp-level primitives (__shfl_down_sync) for final stages.
// It computes the maximum and the sum of exponentials in two reduction phases, then computes log-softmax in a numerically stable way.

template <typename scalar_t>
__global__ void shared_warp_logsoftmax_kernel(const scalar_t* __restrict__ input,
                                               scalar_t* __restrict__ output,
                                               int dim_size) {
    extern __shared__ scalar_t shared[];  // Shared memory for warp-level reduction
    const int warpSize = 32;
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Phase 1: Compute maximum value using per-thread reduction
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = tid; i < dim_size; i += blockDim.x) {
        local_max = max(local_max, input_row[i]);
    }

    unsigned int mask = 0xffffffff;
    // Intra-warp reduction using warp shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(mask, local_max, offset));
    }

    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    if (lane == 0) {
        shared[warp_id] = local_max;
    }
    __syncthreads();

    // Let first warp reduce the partial maximums
    scalar_t block_max = -std::numeric_limits<scalar_t>::infinity();
    int warp_count = (blockDim.x + warpSize - 1) / warpSize;
    if (tid < warp_count) {
        block_max = shared[tid];
    }
    if (tid < warpSize) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_max = max(block_max, __shfl_down_sync(mask, block_max, offset));
        }
    }
    if (tid == 0) {
        shared[0] = block_max;
    }
    __syncthreads();
    scalar_t max_val = shared[0];

    // Phase 2: Compute sum of exp(x - max) using similar reduction
    scalar_t local_sum = 0;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        local_sum += exp(input_row[i] - max_val);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }
    if (lane == 0) {
        shared[warp_id] = local_sum;
    }
    __syncthreads();
    scalar_t block_sum = 0;
    if (tid < warp_count) {
        block_sum = shared[tid];
    }
    if (tid < warpSize) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
    }
    if (tid == 0) {
        shared[0] = block_sum;
    }
    __syncthreads();
    scalar_t sum_val = shared[0];
    scalar_t log_sum = log(sum_val);

    // Phase 3: Compute final log-softmax output
    for (int i = tid; i < dim_size; i += blockDim.x) {
        output_row[i] = (input_row[i] - max_val) - log_sum;
    }
}


torch::Tensor shared_warp_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
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

    int blockSize = 256;
    int warpCount = (blockSize + 31) / 32;
    size_t shared_mem = warpCount * ((input.scalar_type() == torch::kFloat64) ? sizeof(double) : sizeof(float));

    const int blocks = batch_size;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "shared_warp_logsoftmax_cuda_forward", ([&] {
        shared_warp_logsoftmax_kernel<scalar_t><<<blocks, blockSize, shared_mem>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size);
    }));

    // Inverse permutation to restore original layout
    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_warp_logsoftmax_cuda_forward, "Shared & Warp-Level Reduction LogSoftmax forward (CUDA)");
}
