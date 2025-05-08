#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void warp_aligned_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Align to warp size (32 threads)
    constexpr int WARP_SIZE = 32;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Calculate indices ensuring warp-aligned memory access
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    const int num_warps = gridDim.x * warps_per_block;
    
    // Process multiple elements per thread for better memory coalescing
    for (int base_idx = global_warp_id; base_idx < outer_size * (inner_size / WARP_SIZE); base_idx += num_warps) {
        const int outer_idx = base_idx / (inner_size / WARP_SIZE);
        const int inner_base = (base_idx % (inner_size / WARP_SIZE)) * WARP_SIZE;
        
        if (outer_idx >= outer_size) continue;
        
        const int inner_idx = inner_base + lane_id;
        if (inner_idx >= inner_size) continue;
        
        // Calculate aligned offset for coalesced memory access
        const int64_t aligned_offset = outer_idx * dim_size * inner_size + inner_idx;
        
        // Load first element
        scalar_t max_val = input[aligned_offset];
        
        // Reduce along dimension with coalesced access pattern
        #pragma unroll 4
        scalar_t val; for (int i = 1; i < dim_size; i++) { val = input[aligned_offset + i * inner_size];
            const scalar_t val = input[aligned_offset + i * inner_size];
            max_val = max(max_val, val);
        }
        
        // Write result with coalesced access
        output[outer_idx * inner_size + inner_idx] = max_val;
    }
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }
    
    const int64_t dim_size = input.size(dim);
    
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    // Use multiple of warp size for thread block
    const int threads_per_block = 256; // 8 warps per block
    const int num_blocks = (outer_size * ((inner_size + 31) / 32) + (threads_per_block / 32) - 1) / (threads_per_block / 32);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        warp_aligned_max_reduce_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Warp-aligned Max Reduction Forward (CUDA)");
}