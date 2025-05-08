#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// This kernel uses shared memory and warp shuffle reduction to optimize performance.

#include <cooperative_groups.h>

template <typename scalar_t>
__global__ void log_softmax_forward_kernel_shared_mem(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Each block processes one row (batch element)
    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warpSize = 32;
    const int warpId = tid / warpSize;
    const int laneId = tid % warpSize;

    // Shared memory allocation with double buffering
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(smem);

    // Step 1: Compute the maximum value in the row using shared memory for intra-block communication
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = tid; i < dim_size; i += blockSize) {
        scalar_t val = input_row[i];
        local_max = (val > local_max) ? val : local_max;
    }

    // Store local max in shared memory
    shared_data[tid] = local_max;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    scalar_t global_max = shared_data[0];

    // Step 2: Compute the sum of exp(val - global_max) along the row
    scalar_t local_sum = 0;
    for (int i = tid; i < dim_size; i += blockSize) {
        scalar_t exp_val = exp(input_row[i] - global_max);
        local_sum += exp_val;
        output_row[i] = exp_val;  // store intermediate result
    }

    // Store local sums in shared memory
    shared_data[tid] = local_sum;
    __syncthreads();

    // Reduce sum in shared memory
    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    scalar_t global_sum = shared_data[0];

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

    // Choose number of threads: next power of two of dim_size, capped at 1024
    int threads = 1;
    while (threads < dim_size) threads <<= 1;
    if (threads > 1024) threads = 1024;

    // Compute required shared memory
    size_t shared_mem_size = threads * sizeof(float);  // temporary using float size

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda_shared_mem", ([&] {
        shared_mem_size = threads * sizeof(scalar_t);
        log_softmax_forward_kernel_shared_mem<scalar_t><<<batch_size, threads, shared_mem_size>>>(
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
    m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward shared memory (CUDA)");
}
