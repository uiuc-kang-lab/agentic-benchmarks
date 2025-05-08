#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int VEC_SIZE>
__global__ void warp_reduce_mean_optimized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {
    
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;
    
    int output_idx = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;
    if (output_idx >= outer_size * inner_size) return;
    
    int outer_idx = output_idx / inner_size;
    int inner_idx = output_idx % inner_size;
    int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    using VecType = typename std::conditional<VEC_SIZE == 4, float4, float2>::type;
    VecType vec_val;
    scalar_t sum = 0;
    
    // Vectorized loading with aligned accesses
    for (int i = threadIdx.x * VEC_SIZE; i < dim_size; i += BLOCK_SIZE * VEC_SIZE) {
        if (i + VEC_SIZE <= dim_size) {
            reinterpret_cast<VecType&>(vec_val) = *reinterpret_cast<const VecType*>(&input[input_offset + i * inner_size]);
            #pragma unroll
            for (int v = 0; v < VEC_SIZE; v++)
                sum += reinterpret_cast<scalar_t*>(&vec_val)[v];
        } else { // Handle remaining elements
            #pragma unroll
            for (int v = 0; v < VEC_SIZE && (i + v) < dim_size; v++)
                sum += input[input_offset + (i + v) * inner_size];
        }
    }

    // Warp-level reduction with no diverging branches
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Single write per warp with coalesced access
    if (threadIdx.x == 0)
        output[output_idx] = sum / dim_size;
}

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= sizes[i];
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) inner_size *= sizes[i];
    
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    constexpr int VEC_SIZE = 4;
    constexpr int WARPS_PER_BLOCK = 8;
    dim3 block(WARPS_PER_BLOCK, 32);
    int grid_x = (outer_size * inner_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", [&] {
        warp_reduce_mean_optimized_kernel<scalar_t, VEC_SIZE><<<grid_x, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Optimized mean reduction with warp shuffles");
}