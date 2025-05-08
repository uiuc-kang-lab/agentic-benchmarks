#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// This kernel uses shared memory and warp shuffle reduction to optimize reductions.

template <typename scalar_t>
__global__ void log_softmax_forward_kernel_optimized(
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

    // Step 1: Compute the maximum value in the row using shared memory and warp shuffle reduction
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = tid; i < dim_size; i += blockSize) {
        scalar_t val = input_row[i];
        local_max = (val > local_max) ? val : local_max;
    }

    // Use shared memory for intra-block reduction
    extern __shared__ scalar_t shared_data[];
    shared_data[tid] = local_max;
    __syncthreads();

    // Reduce within block using shared memory
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    // Final reduction using warp shuffle
    local_max = shared_data[0];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(mask, local_max, offset));
    }

    // Step 2: Compute the sum of exp(val - local_max) using shared memory and warp shuffle reduction
    scalar_t local_sum = 0;
    for (int i = tid; i < dim_size; i += blockSize) {
        scalar_t exp_val = exp(input_row[i] - local_max);
        local_sum += exp_val;
        output_row[i] = exp_val;  // store intermediate result
    }

    // Use shared memory for intra-block reduction
    shared_data[tid] = local_sum;
    __syncthreads();

    // Reduce within block using shared memory
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Final reduction using warp shuffle
    local_sum = shared_data[0];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    scalar_t log_sum = log(local_sum);

    // Step 3: Compute final log softmax output
    for (int i = tid; i < dim_size; i += blockSize) {
        output_row[i] = (input_row[i] - local_max) - log_sum;
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

    // Choose number of threads: next power of two of dim_size, capped at 1024
    int threads = 1;
    while (threads < dim_size) threads <<= 1;
    if (threads > 1024) threads = 1024;

    // Compute required shared memory: one scalar per thread
    size_t shared_mem_size = threads * sizeof(float);  // temporary using float size

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda_optimized", ([&] {
        shared_mem_size = threads * sizeof(scalar_t);
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
    m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward optimized (CUDA)");
}

// runtime: 0.01000 milliseconds
// speedup over torch: 1.00x
