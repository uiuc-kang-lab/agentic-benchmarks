#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Utility function: compute the next highest power of two (host-side)
static inline int next_power_of_two(int x) {
    int power = 1;
    while (power < x) {
        power <<= 1;
    }
    return power;
}

// Templated CUDA kernel for LogSoftmax with tunable block size
template <typename scalar_t, int BLOCK_SIZE>
__global__ void blocksize_tuning_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    // Each block handles one row (batch element)
    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Allocate shared memory dynamically
    extern __shared__ scalar_t sdata[];

    // Phase 1: Compute the maximum value in the row
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        local_max = max(local_max, input_row[idx]);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Reduction for maximum in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    scalar_t max_val = sdata[0];
    
    // Phase 2: Compute the sum of exp(x - max_val) for numerical stability
    scalar_t local_sum = 0;
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        local_sum += exp(input_row[idx] - max_val);
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    
    // Reduction for summing exp values
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    scalar_t log_sum = log(sdata[0]);
    
    // Phase 3: Write back the LogSoftmax results
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        output_row[idx] = (input_row[idx] - max_val) - log_sum;
    }
}

// Host function to prepare and launch the kernel
torch::Tensor blocksize_tuning_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
                "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = (dim >= 0) ? dim : dim + ndim;

    // Permute input so that the target dimension becomes the last dimension
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

    // Dynamically select the optimal block size based on dim_size
    int candidate = next_power_of_two(dim_size);
    candidate = candidate < 32 ? 32 : candidate;
    candidate = candidate > 512 ? 512 : candidate;
    int optimal_block_size = candidate;  // Candidate is one of {32, 64, 128, 256, 512}

    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "blocksize_tuning_logsoftmax_cuda_forward", ([&] {
        if (optimal_block_size == 32) {
            blocksize_tuning_logsoftmax_kernel<scalar_t, 32><<<blocks, 32, 32 * sizeof(scalar_t)>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 64) {
            blocksize_tuning_logsoftmax_kernel<scalar_t, 64><<<blocks, 64, 64 * sizeof(scalar_t)>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 128) {
            blocksize_tuning_logsoftmax_kernel<scalar_t, 128><<<blocks, 128, 128 * sizeof(scalar_t)>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 256) {
            blocksize_tuning_logsoftmax_kernel<scalar_t, 256><<<blocks, 256, 256 * sizeof(scalar_t)>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 512) {
            blocksize_tuning_logsoftmax_kernel<scalar_t, 512><<<blocks, 512, 512 * sizeof(scalar_t)>>>(
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
    m.def("forward", &blocksize_tuning_logsoftmax_cuda_forward, "Blocksize Tuning LogSoftmax forward (CUDA)");
}
