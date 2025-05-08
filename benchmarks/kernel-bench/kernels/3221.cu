#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// This kernel uses stride loops to efficiently handle large workloads,
// ensuring that each thread correctly processes elements beyond its initial allocation.

template <typename scalar_t>
__global__ void log_softmax_forward_kernel_stride(
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

    // Step 1: Compute the maximum value in the row using stride loop
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = tid; i < dim_size; i += blockSize) {
        scalar_t val = input_row[i];
        local_max = max(val, local_max);
    }

    // Warp-level reduction for maximum using shfl_down_sync
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t temp = __shfl_down_sync(mask, local_max, offset);
        local_max = max(temp, local_max);
    }

    // Shared memory for warp reduction results
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(smem);
    
    // Inter-warp reduction using first thread of each warp
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    if (lane_id == 0) {
        shared_data[warp_id] = local_max;
    }
    __syncthreads();

    // Final reduction using single warp
    scalar_t global_max = -std::numeric_limits<scalar_t>::infinity();
    if (tid < (blockSize + warpSize - 1) / warpSize) {
        global_max = shared_data[tid];
    }
    
    // Warp-level reduction for the final result
    for (int offset = (blockSize + warpSize - 1) / (2 * warpSize); offset > 0; offset /= 2) {
        scalar_t temp = __shfl_down_sync(mask, global_max, offset);
        if (tid < offset) {
            global_max = max(global_max, temp);
        }
    }
    
    // Broadcast result to all threads
    global_max = __shfl_sync(mask, global_max, 0);

    // Step 2: Compute the sum of exp(val - global_max) using stride loop
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
        shared_data[tid / warpSize] = local_sum;
    }
    __syncthreads();

    scalar_t global_sum;
    if (tid == 0) {
        global_sum = shared_data[0];
        int num_warps = (blockSize + warpSize - 1) / warpSize;
        for (int i = 1; i < num_warps; ++i) {
            global_sum += shared_data[i];
        }
        shared_data[0] = log(global_sum);
    }
    __syncthreads();

    scalar_t log_sum = shared_data[0];

    // Step 3: Compute final log softmax output using stride loop
    for (int i = tid; i < dim_size; i += blockSize) {
        output_row[i] = (input_row[i] - global_max) - log_sum;
    }
}

// Host function launching the kernel
torch::Tensor log_softmax_cuda_forward_stride(torch::Tensor input, int64_t dim) {
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

    // Compute required shared memory: one scalar per warp
    int warpSize = 32;
    int num_warps = (threads + warpSize - 1) / warpSize;
    size_t shared_mem_size = num_warps * sizeof(float);  // temporary using float size

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda_stride", ([&] {
        shared_mem_size = num_warps * sizeof(scalar_t);
        log_softmax_forward_kernel_stride<scalar_t><<<batch_size, threads, shared_mem_size>>>(
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
    m.def("forward", &log_softmax_cuda_forward_stride, "LogSoftmax forward stride (CUDA)");
}