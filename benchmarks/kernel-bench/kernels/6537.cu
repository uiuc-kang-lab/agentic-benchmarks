#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void warp_reduce_mean_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {
    
    // Warp-based reduction with 32 threads per output
    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_ROWS = 4;  // Number of warps per block
    
    // Shared memory for partial sums
    __shared__ scalar_t shared_data[BLOCK_ROWS][WARP_SIZE];
    
    // 2D grid: x-dim threads for reduction, y-dim for outputs
    int output_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (output_idx >= outer_size * inner_size) return;
    
    int outer_idx = output_idx / inner_size;
    int inner_idx = output_idx % inner_size;
    int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    scalar_t sum = 0;
    
    // Parallel reduction across warp threads with sequential loads to avoid memory divergence
    #pragma unroll 4
    for (int i = threadIdx.x; i < dim_size; i += WARP_SIZE) {
        sum += input[input_offset + i * inner_size];
    }
    
    // Store partial sum in shared memory (avoiding bank conflicts by using warp index as offset)
    shared_data[threadIdx.y][threadIdx.x] = sum;
    __syncwarp();
    
    // Warp-level reduction using shfl_down with improved ILP
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Write final result with first thread of warp
    if (threadIdx.x == 0) {
        output[output_idx] = sum / static_cast<scalar_t>(dim_size);
    }
}

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }
    
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());
    
    // Configure 2D blocks with 32x4 threads (4 outputs per block)
    const int WARPS_PER_BLOCK = 4;
    dim3 block(32, WARPS_PER_BLOCK);
    int grid_x = (outer_size * inner_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(grid_x, 1);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "warp_reduce_mean_cuda", ([&] {
        warp_reduce_mean_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean reduction with warp shuffles (CUDA)");
}