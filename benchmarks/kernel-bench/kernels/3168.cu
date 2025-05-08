#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Kernel using CUDA streams to overlap computation and memory transfers

template <typename scalar_t, int BLOCK_SIZE>
__global__ void streamed_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    __shared__ scalar_t sdata[BLOCK_SIZE];

    // Phase 1: Compute the maximum value in the row using warp-level reduction
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        scalar_t val = input_row[idx];
        local_max = max(local_max, val);
    }

    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(mask, local_max, offset);
        local_max = max(local_max, other);
    }

    int warp_id = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    if (lane == 0) {
        sdata[warp_id] = local_max;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        scalar_t block_max = -std::numeric_limits<scalar_t>::infinity();
        for (int i = 0; i < BLOCK_SIZE / warpSize; i++) {
            block_max = max(block_max, sdata[i]);
        }
        sdata[0] = block_max;
    }
    __syncthreads();
    scalar_t max_val = sdata[0];

    // Phase 2: Compute the sum of exp(x - max_val) using warp-level reduction
    scalar_t local_sum = 0;
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        scalar_t exp_val = exp(input_row[idx] - max_val);
        local_sum += exp_val;
    }

    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    if (lane == 0) {
        atomicAdd(&sdata[0], local_sum);
    }
    __syncthreads();
    scalar_t sum = sdata[0];
    scalar_t log_sum = log(sum);

    // Phase 3: Write back the final LogSoftmax values
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        output_row[idx] = (input_row[idx] - max_val) - log_sum;
    }
}

// Host function using CUDA streams

torch::Tensor streamed_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
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

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "streamed_logsoftmax_cuda_forward", ([&] {
        if (optimal_block_size == 32) {
            streamed_logsoftmax_kernel<scalar_t, 32><<<blocks, 32, 0, stream1>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 64) {
            streamed_logsoftmax_kernel<scalar_t, 64><<<blocks, 64, 0, stream2>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 128) {
            streamed_logsoftmax_kernel<scalar_t, 128><<<blocks, 128, 0, stream1>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 256) {
            streamed_logsoftmax_kernel<scalar_t, 256><<<blocks, 256, 0, stream2>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 512) {
            streamed_logsoftmax_kernel<scalar_t, 512><<<blocks, 512, 0, stream1>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        }
    }));

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &streamed_logsoftmax_cuda_forward, "Streamed LogSoftmax forward (CUDA)");
}
