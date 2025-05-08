#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for warp-level reduction using shfl_down
template <typename scalar_t>
__device__ scalar_t warp_reduce_sum(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Improved Kernel function for mean reduction using better thread and block indexing
template <typename scalar_t>
__global__ void optimized_warp_reduce_mean_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {
    
    constexpr int WARP_SIZE = 32;
    constexpr int OUTPUTS_PER_BLOCK = 8;

    int block_start_output_idx = blockIdx.x * OUTPUTS_PER_BLOCK;
    int output_idx = block_start_output_idx + threadIdx.y;

    if (output_idx >= outer_size * inner_size) return;
    
    int outer_idx = output_idx / inner_size;
    int inner_idx = output_idx % inner_size;
    int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    scalar_t sum = 0;
    for (int i = threadIdx.x; i < dim_size; i += WARP_SIZE) {
        sum += input[input_offset + i * inner_size];
    }
    
    sum = warp_reduce_sum(sum);
    
    if (threadIdx.x == 0) {
        output[output_idx] = sum / dim_size;
    }
}

// Host function to launch the kernel
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
    
    const int WARPS_PER_BLOCK = 8;  // Using more warps per block for better occupancy
    dim3 block(32, WARPS_PER_BLOCK);
    // Calculate grid size based on OUTPUTS_PER_BLOCK to match kernel configuration
int grid_x = (outer_size * inner_size + OUTPUTS_PER_BLOCK - 1) / OUTPUTS_PER_BLOCK;
    dim3 grid(grid_x, 1);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_warp_reduce_mean_cuda", ([&] {
        optimized_warp_reduce_mean_kernel<scalar_t><<<grid, block>>>(
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
    m.def("forward", &mean_reduce_cuda, "Optimized mean reduction with warp shuffles (CUDA)");
}