#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t numel,
    int64_t reduce_size,
    int64_t outer_size,
    int64_t inner_size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_size = warpSize;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;
    
    // Calculate the number of elements each warp processes
    const int64_t elements_per_warp = (outer_size * inner_size + gridDim.x * blockDim.x / warp_size - 1) 
                                     / (gridDim.x * blockDim.x / warp_size);
    const int64_t warp_start = warp_id * elements_per_warp;
    const int64_t warp_end = min(warp_start + elements_per_warp, outer_size * inner_size);
    
    // Process elements assigned to this warp
    for (int64_t idx = warp_start + lane_id; idx < warp_end; idx += warp_size) {
        if (idx < outer_size * inner_size) {
            const int64_t outer_idx = idx / inner_size;
            const int64_t inner_idx = idx % inner_size;
            
            scalar_t sum = 0;
            const int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;
            
            // Coalesced memory access pattern
            #pragma unroll 4
            for (int i = 0; i < reduce_size; i++) {
                sum += input[base_idx + i * inner_size];
            }
            
            output[outer_idx * inner_size + inner_idx] = sum;
        }
    }
}

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }
    
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());
    
    const int threads_per_block = 256;
    const int blocks = (outer_size * inner_size + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        sum_reduce_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel(),
            reduce_size,
            outer_size,
            inner_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA)");
}