#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t>
__global__ void grid_optimized_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim_size) {

    // 2D grid configuration for better hardware utilization
    const int batch_idx = blockIdx.y * gridDim.x + blockIdx.x;
    if (batch_idx >= batch_size) return;

    // Use shared memory for reductions
    extern __shared__ scalar_t shared_mem[];
    scalar_t* smax = shared_mem;
    scalar_t* ssum = &shared_mem[blockDim.x];

    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Compute max value using grid-strided loop
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    
    #pragma unroll 4
    for (int idx = threadIdx.x; idx < dim_size; idx += blockDim.x) {
        thread_max = max(thread_max, input_row[idx]);
    }

    // Warp-level reduction for maximum
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        scalar_t other = __shfl_xor_sync(0xffffffff, thread_max, offset);
        thread_max = max(thread_max, other);
    }

    // Block-level reduction for maximum
    if (threadIdx.x % warpSize == 0) {
        smax[threadIdx.x / warpSize] = thread_max;
    }
    __syncthreads();

    if (threadIdx.x < warpSize) {
        scalar_t warp_max = (threadIdx.x < (blockDim.x / warpSize)) ? 
            smax[threadIdx.x] : -std::numeric_limits<scalar_t>::infinity();
        
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            scalar_t other = __shfl_xor_sync(0xffffffff, warp_max, offset);
            warp_max = max(warp_max, other);
        }
        
        if (threadIdx.x == 0) {
            smax[0] = warp_max;
        }
    }
    __syncthreads();

    const scalar_t max_val = smax[0];

    // Compute sum of exponentials using grid-strided loop
    scalar_t thread_sum = 0;
    
    #pragma unroll 4
    for (int idx = threadIdx.x; idx < dim_size; idx += blockDim.x) {
        const scalar_t val = exp(input_row[idx] - max_val);
        thread_sum += val;
        // Store intermediate result for final computation
        output_row[idx] = val;
    }

    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, offset);
    }

    // Block-level reduction for sum
    if (threadIdx.x % warpSize == 0) {
        ssum[threadIdx.x / warpSize] = thread_sum;
    }
    __syncthreads();

    if (threadIdx.x < warpSize) {
        scalar_t warp_sum = (threadIdx.x < (blockDim.x / warpSize)) ? 
            ssum[threadIdx.x] : 0;
        
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, offset);
        }
        
        if (threadIdx.x == 0) {
            ssum[0] = warp_sum;
        }
    }
    __syncthreads();

    const scalar_t log_sum = log(ssum[0]);

    // Compute final values using grid-strided loop
    #pragma unroll 4
    for (int idx = threadIdx.x; idx < dim_size; idx += blockDim.x) {
        output_row[idx] = log(output_row[idx]) - log_sum;
    }
}

torch::Tensor grid_optimized_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
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

    // Optimize grid dimensions for H100
    const int threads_per_block = 256;
    const int max_blocks_per_sm = 16;
    const int num_sm = 132; // H100 has 132 SMs

    // Calculate 2D grid dimensions
    dim3 grid;
    grid.x = min(max_blocks_per_sm * num_sm, (int)ceil(batch_size / 32.0));
    grid.y = (batch_size + grid.x - 1) / grid.x;
    
    const size_t shared_mem_size = 2 * threads_per_block * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_optimized_logsoftmax_cuda_forward", ([&] {
        grid_optimized_logsoftmax_kernel<scalar_t><<<grid, threads_per_block, shared_mem_size>>>(
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
    m.def("forward", &grid_optimized_logsoftmax_cuda_forward, "Grid Optimized LogSoftmax forward (CUDA)");
}