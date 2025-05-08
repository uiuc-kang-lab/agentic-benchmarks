#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// This kernel combines shared memory reduction and warp-level operations
// to minimize __syncthreads() calls and leverage warp shuffle operations.

template <typename scalar_t>
__global__ void log_softmax_forward_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    // Each block handles one row (batch element)
    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    const int tid = threadIdx.x;
    const int warpSize = 32;
    const unsigned int mask = 0xffffffff;

    // Step 1: Compute the maximum value in the row using warp shuffle reduction
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = tid; i < dim_size; i += blockDim.x) {
        scalar_t val = input_row[i];
        local_max = (val > local_max) ? val : local_max;
    }

    // Reduce within warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(mask, local_max, offset));
    }

    // Allocate shared memory for per-warp maxima
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(smem);

    if (threadIdx.x % warpSize == 0) {
        shared_data[threadIdx.x / warpSize] = local_max;
    }

    __syncthreads();

    // Final reduction within block
    if (threadIdx.x < warpSize) {
        local_max = shared_data[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local_max = max(local_max, __shfl_down_sync(mask, local_max, offset));
        }
    }

    if (threadIdx.x == 0) shared_data[0] = local_max;
    __syncthreads();
    local_max = shared_data[0];

    // Step 2: Compute sum of exp(val - local_max) using warp-level reduction
    scalar_t local_sum = 0;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        scalar_t exp_val = exp(input_row[i] - local_max);
        output_row[i] = exp_val;  // store intermediate result
        local_sum += exp_val;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Store sum in shared memory
    if (threadIdx.x % warpSize == 0) {
        shared_data[threadIdx.x / warpSize] = local_sum;
    }
    __syncthreads();  // Gather warp sums

    // Final sum reduction by first warp
    scalar_t global_sum = 0;
    if (threadIdx.x < warpSize) {
        if (threadIdx.x == 0) global_sum = shared_data[0];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            global_sum += __shfl_down_sync(mask, global_sum, offset);
        }
    }

    if (threadIdx.x == 0) shared_data[0] = global_sum;
    __syncthreads();
    global_sum = shared_data[0];

    scalar_t log_sum = log(global_sum);

    // Step 3: Compute final log softmax output
    for (int i = tid; i < dim_size; i += blockDim.x) {
        output_row[i] = (input_row[i] - local_max) - log_sum;
    }
}

// Host function launching the kernel
torch::Tensor log_softmax_cuda_forward_optimized(torch::Tensor input, int64_t dim) {
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

    // Choose number of threads: next power of two of dim_size, capped at 1024
    int threads = (dim_size + 31) / 32 * 32; // Round up to the next multiple of 32
    if (threads > 1024) threads = 1024;

    // Compute required shared memory
    int warpSize = 32;
    int num_warps = (threads + warpSize - 1) / warpSize;
    size_t shared_mem_size = num_warps * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda_optimized", ([&] {
        log_softmax_forward_kernel_optimized<scalar_t><<<batch_size, threads, shared_mem_size>>>(
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
    m.def("forward", &log_softmax_cuda_forward_optimized, "LogSoftmax forward optimized (CUDA)");
}
