#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Modular kernel with device functions for readability and maintainability

// Device function to compute the maximum value in a row
__device__ float compute_max(const float* __restrict__ row, int dim_size, int threadIdx, int blockDim) {
    float local_max = -std::numeric_limits<float>::infinity();
    for (int idx = threadIdx; idx < dim_size; idx += blockDim) {
        local_max = max(local_max, row[idx]);
    }
    return local_max;
}

// Device function to compute the sum of exponentials
__device__ float compute_sum_exp(const float* __restrict__ row, float max_val, int dim_size, int threadIdx, int blockDim) {
    float local_sum = 0;
    for (int idx = threadIdx; idx < dim_size; idx += blockDim) {
        local_sum += exp(row[idx] - max_val);
    }
    return local_sum;
}

// Device function to perform warp-level reduction
__device__ float warp_reduce(float val) {
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// Kernel function
__global__ void modular_logsoftmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int dim_size) {

    int batch_idx = blockIdx.x;
    const float* input_row = input + batch_idx * dim_size;
    float* output_row = output + batch_idx * dim_size;

    // Shared memory for reductions
    __shared__ float sdata[32];  // Supports up to 32 warps

    // Phase 1: Compute the maximum value
    float local_max = compute_max(input_row, dim_size, threadIdx.x, blockDim.x);
    local_max = warp_reduce(local_max);

    int warp_id = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    if (lane == 0) {
        sdata[warp_id] = local_max;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_max = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < blockDim.x / warpSize; i++) {
            block_max = max(block_max, sdata[i]);
        }
        sdata[0] = block_max;
    }
    __syncthreads();
    float max_val = sdata[0];

    // Phase 2: Compute the sum of exp(x - max_val)
    float local_sum = compute_sum_exp(input_row, max_val, dim_size, threadIdx.x, blockDim.x);
    local_sum = warp_reduce(local_sum);

    if (lane == 0) {
        atomicAdd(&sdata[0], local_sum);
    }
    __syncthreads();
    float sum = sdata[0];
    float log_sum = log(sum);

    // Phase 3: Write back the final LogSoftmax values
    for (int idx = threadIdx.x; idx < dim_size; idx += blockDim.x) {
        output_row[idx] = (input_row[idx] - max_val) - log_sum;
    }
}

// Host function

torch::Tensor modular_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
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

    modular_logsoftmax_kernel<<<blocks, optimal_block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim_size);

    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &modular_logsoftmax_cuda_forward, "Modular LogSoftmax forward (CUDA)");
}
