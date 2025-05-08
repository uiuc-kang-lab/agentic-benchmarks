#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Unrolled kernel using a compile-time block size parameter for performance tuning
// Supported block sizes: 32, 64, 128, 256, and 512

template <typename scalar_t, int BLOCK_SIZE>
__global__ void unroll_tuned_log_softmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    // Each block handles one row (batch element)
    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Shared memory for reductions
    __shared__ scalar_t sdata[BLOCK_SIZE];

    // Phase 1: Compute the maximum value in the row
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        scalar_t val = input_row[idx];
        local_max = (val > local_max) ? val : local_max;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Reduction to find max value
    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] = (sdata[threadIdx.x] > sdata[threadIdx.x + stride]) ? 
                                   sdata[threadIdx.x] : sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    scalar_t max_val = sdata[0];
    __syncthreads();

    // Phase 2: Compute the sum of exp(x - max_val) for numerical stability
    scalar_t local_sum = 0;
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        // Compute exponentials
        scalar_t exp_val = exp(input_row[idx] - max_val);
        local_sum += exp_val;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduction to compute total sum
    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    scalar_t sum = sdata[0];
    scalar_t log_sum = log(sum);
    __syncthreads();

    // Phase 3: Write back the final LogSoftmax values
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        output_row[idx] = (input_row[idx] - max_val) - log_sum;
    }
}


// Host function
// This function permutes the input so that the reduction occurs on the last dimension,
// selects an optimal block size from the set {32, 64, 128, 256, 512} based on dim_size,
// and then launches the tuned CUDA kernel.

torch::Tensor unroll_tuned_log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
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

    // Select an optimal block size from {32, 64, 128, 256, 512}
    int optimal_block_size = 256; // Default value
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
        optimal_block_size = 512; // For larger dimensions, cap at 512 threads per block
    }

    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "unroll_tuned_log_softmax_forward_cuda", ([&] {
        if (optimal_block_size == 32) {
            unroll_tuned_log_softmax_forward_kernel<scalar_t, 32><<<blocks, 32>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 64) {
            unroll_tuned_log_softmax_forward_kernel<scalar_t, 64><<<blocks, 64>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 128) {
            unroll_tuned_log_softmax_forward_kernel<scalar_t, 128><<<blocks, 128>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 256) {
            unroll_tuned_log_softmax_forward_kernel<scalar_t, 256><<<blocks, 256>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 512) {
            unroll_tuned_log_softmax_forward_kernel<scalar_t, 512><<<blocks, 512>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        }
    }));

    // Inverse permutation to restore original data layout
    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &unroll_tuned_log_softmax_cuda_forward, "Unroll Tuned LogSoftmax forward (CUDA)");
}
