#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

template <typename scalar_t>
__global__ void log_softmax_forward_kernel_grid(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim_size) {

    // Use 2D grid for better work distribution
    const int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size) return;

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warpSize = 32;
    const unsigned int mask = 0xffffffff;

    // Calculate input/output row pointers
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Shared memory for reductions
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(smem);

    // Step 1: Find maximum value
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    #pragma unroll 4
    for (int i = tid; i < dim_size; i += blockSize) {
        local_max = max(local_max, input_row[i]);
    }

    // Warp-level reduction for maximum
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        scalar_t temp = __shfl_down_sync(mask, local_max, offset);
        local_max = max(local_max, temp);
    }

    // Store warp results
    const int warp_id = tid / warpSize;
    if (tid % warpSize == 0) {
        shared_data[warp_id * 2] = local_max;
    }
    __syncthreads();

    // Final reduction for maximum
    if (tid == 0) {
        scalar_t block_max = shared_data[0];
        const int num_warps = (blockSize + warpSize - 1) / warpSize;
        #pragma unroll
        for (int i = 1; i < num_warps; i++) {
            block_max = max(block_max, shared_data[i]);
        }
        shared_data[0] = block_max;
    }
    __syncthreads();
    const scalar_t max_val = shared_data[0];

    // Step 2: Compute sum of exponentials
    scalar_t local_sum = 0;
    #pragma unroll 4
    for (int i = tid; i < dim_size; i += blockSize) {
        const scalar_t exp_val = exp(input_row[i] - max_val);
        output_row[i] = exp_val;  // Store for later use
        local_sum += exp_val;
    }

    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    if (tid % warpSize == 0) {
        shared_data[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction for sum
    if (tid == 0) {
        scalar_t block_sum = shared_data[0];
        const int num_warps = (blockSize + warpSize - 1) / warpSize;
        #pragma unroll
        for (int i = 1; i < num_warps; i++) {
            block_sum += shared_data[i];
        }
        shared_data[0] = log(block_sum);
    }
    __syncthreads();
    const scalar_t log_sum = shared_data[0];

    // Step 3: Compute final output
    #pragma unroll 4
    for (int i = tid; i < dim_size; i += blockSize) {
        output_row[i] = input_row[i] - max_val - log_sum;
    }
}

torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = (dim >= 0) ? dim : dim + ndim;

    std::vector<int64_t> permute_dims;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) permute_dims.push_back(i);
    }
    permute_dims.push_back(dim);
    
    input = input.permute(permute_dims).contiguous();
    auto output = torch::empty_like(input);

    const int64_t batch_size = input.numel() / input.size(-1);
    const int64_t dim_size = input.size(-1);

    // Optimize thread block configuration
    const int threads_x = 256;  // Power of 2, good for warp operations
    const int threads_y = 4;    // Process multiple batches per block
    const int blocks_y = (batch_size + threads_y - 1) / threads_y;

    dim3 threads(threads_x, threads_y);
    dim3 blocks(1, blocks_y);

    // Shared memory size per block
    const int warps_per_block = (threads_x * threads_y + 31) / 32;
    size_t shared_mem_size = warps_per_block * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda_grid", ([&] {
        shared_mem_size = warps_per_block * sizeof(scalar_t);
        log_softmax_forward_kernel_grid<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim_size);
    }));

    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward grid optimized (CUDA)");
}