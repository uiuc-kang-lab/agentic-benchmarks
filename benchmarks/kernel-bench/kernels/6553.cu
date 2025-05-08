#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for frequently accessed dimension information
__constant__ int64_t c_dim_size;
__constant__ int64_t c_inner_size;
__constant__ int64_t c_outer_size;

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void warp_reduce_mean_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output) {
    
    constexpr int WARP_SIZE = 32;
    int output_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (output_idx >= c_outer_size * c_inner_size) return;
    
    int outer_idx = output_idx / c_inner_size;
    int inner_idx = output_idx % c_inner_size;
    int input_offset = outer_idx * c_dim_size * c_inner_size + inner_idx;
    
    // Pre-calculate stride for better memory access pattern
    const int stride = c_inner_size;
    
    scalar_t sum = 0;
    #pragma unroll 4
    for (int i = threadIdx.x; i < c_dim_size; i += WARP_SIZE) {
        sum = __fmaf_rn(1.0f, input[input_offset + i * stride], sum);
    }
    
    // Perform warp-level reduction
    sum = warp_reduce_sum(sum);
    
    if (threadIdx.x == 0) {
        // Use fast division approximation for better performance
        output[output_idx] = __fdividef(sum, static_cast<scalar_t>(c_dim_size));
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
    
    // Copy dimension information to constant memory
    cudaMemcpyToSymbol(c_dim_size, &dim_size, sizeof(int64_t));
    cudaMemcpyToSymbol(c_inner_size, &inner_size, sizeof(int64_t));
    cudaMemcpyToSymbol(c_outer_size, &outer_size, sizeof(int64_t));
    
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());
    
    const int WARPS_PER_BLOCK = 8;  // Increased warps per block for better occupancy
    dim3 block(32, WARPS_PER_BLOCK);
    int grid_x = (outer_size * inner_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(grid_x, 1);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "warp_reduce_mean_cuda", ([&] {
        warp_reduce_mean_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean reduction with constant memory (CUDA)");
}