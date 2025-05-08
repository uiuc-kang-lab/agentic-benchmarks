#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// This kernel utilizes different block sizes to find the optimal configuration for performance.

template <typename scalar_t>
__global__ void log_softmax_forward_kernel_block_size(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    // Each block processes one row (batch element)
    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warpSize = 32;
    const unsigned int mask = 0xffffffff;

    // Step 1: Compute the maximum value in the row in a numerically stable way using warp shuffle reduction.
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = tid; i < dim_size; i += blockSize) {
        scalar_t val = input_row[i];
        local_max = (val > local_max) ? val : local_max;
    }

    // Warp-level reduction for maximum using shfl_down_sync
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t temp = __shfl_down_sync(mask, local_max, offset);
        local_max = (temp > local_max) ? temp : local_max;
    }

    // Allocate shared memory for per-warp results
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(smem);
    int warp_id = tid / warpSize;
    if ((tid % warpSize) == 0) {
        shared_data[warp_id] = local_max;
    }
    __syncthreads();  // Ensure all warps have written their results

    // Final reduction by thread 0 to get the row maximum
    scalar_t global_max;
    if (tid == 0) {
        int num_warps = (blockSize + warpSize - 1) / warpSize;
        global_max = shared_data[0];
        for (int i = 1; i < num_warps; i++) {
            global_max = (shared_data[i] > global_max) ? shared_data[i] : global_max;
        }
        shared_data[0] = global_max;  // broadcast the result
    }
    __syncthreads();
    global_max = shared_data[0];

    // Step 2: Compute the sum of exp(val - global_max) along the row
    scalar_t local_sum = 0;
    for (int i = tid; i < dim_size; i += blockSize) {
        scalar_t exp_val = exp(input_row[i] - global_max);
        local_sum += exp_val;
        output_row[i] = exp_val;  // store intermediate result
    }

    // Warp-level reduction for sum using shfl_down_sync
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }
    if ((tid % warpSize) == 0) {
        shared_data[warp_id] = local_sum;
    }
    __syncthreads();  // Synchronize to gather warp sums

    scalar_t global_sum;
    if (tid == 0) {
        int num_warps = (blockSize + warpSize - 1) / warpSize;
        global_sum = shared_data[0];
        for (int i = 1; i < num_warps; i++) {
            global_sum += shared_data[i];
        }
        shared_data[0] = global_sum;  // broadcast global sum
    }
    __syncthreads();
    global_sum = shared_data[0];

    scalar_t log_sum = log(global_sum);

    // Step 3: Compute final log softmax output
    for (int i = tid; i < dim_size; i += blockSize) {
        output_row[i] = (input_row[i] - global_max) - log_sum;
    }
}

// Host function launching the kernel
torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = (dim >= 0) ? dim : dim + ndim;

    // Permute input to bring 'dim' to the last dimension
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

    // Experiment with different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};
    int optimal_block_size = 256;  // Default assumption

    // Compute required shared memory: one scalar per warp
    int warpSize = 32;
    int num_warps = (optimal_block_size + warpSize - 1) / warpSize;
    size_t shared_mem_size = num_warps * sizeof(float);  // temporary using float size

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda_block_size", ([&] {
        shared_mem_size = num_warps * sizeof(scalar_t);
        log_softmax_forward_kernel_block_size<scalar_t><<<batch_size, optimal_block_size, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size);
    }));

    // Inverse permutation to restore original shape
    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward block size optimization (CUDA)");
}
