#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void warp_aligned_cumprod_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_chains) {

    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;
    
    int chain_id = blockIdx.x;
    if (chain_id >= total_chains) return;
    
    int batch_idx = chain_id / stride;
    int in_idx = chain_id % stride;
    int64_t base = batch_idx * (dim_size * stride) + in_idx;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    __shared__ scalar_t warp_products[WARPS_PER_BLOCK];
    
    int elements_per_warp = (dim_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    elements_per_warp = ((elements_per_warp + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    
    int warp_start = warp_id * elements_per_warp;
    scalar_t warp_product = 1;
    
    #pragma unroll 4
    for (int i = warp_start + lane_id; i < warp_start + elements_per_warp && i < dim_size; i += WARP_SIZE) {
        scalar_t val = input[base + i * stride];
        warp_product *= val;
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_product *= __shfl_down_sync(0xffffffff, warp_product, offset);
    }
    
    if (lane_id == 0) {
        warp_products[warp_id] = warp_product;
    }
    __syncthreads();
    
    if (warp_id == 0 && lane_id < WARPS_PER_BLOCK) {
        scalar_t prefix = 1;
        scalar_t curr = warp_products[lane_id];
        warp_products[lane_id] = prefix;
        prefix *= curr;
    }
    __syncthreads();
    
    scalar_t prefix = (warp_id == 0) ? 1 : warp_products[warp_id - 1];
    scalar_t running_product = prefix;
    
    #pragma unroll 4
    for (int i = warp_start + lane_id; i < warp_start + elements_per_warp && i < dim_size; i += WARP_SIZE) {
        running_product *= input[base + i * stride];
        output[base + i * stride] = running_product;
    }
}

torch::Tensor warp_aligned_cumprod_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride_val = strides[dim];
    int64_t total_chains = input.numel() / dim_size;
    
    const int threads = 256;  // 8 warps per block
    dim3 blocks(total_chains);
    dim3 threads_per_block(threads);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "warp_aligned_cumprod", ([&] {
        warp_aligned_cumprod_kernel<scalar_t><<<blocks, threads_per_block>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            dim_size,
            stride_val,
            total_chains
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_aligned_cumprod_forward, "Warp-aligned cumulative product forward (CUDA)");
}