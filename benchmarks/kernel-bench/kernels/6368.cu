#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void warp_sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t total_output) {

    // Each warp processes multiple elements for better occupancy
    const int warp_size = 32;
    const int warps_per_block = blockDim.x / warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    
    // Calculate global warp index
    const int global_warp_idx = blockIdx.x * warps_per_block + warp_id;
    
    // Each warp handles multiple output elements
    const int elements_per_warp = 4;
    const int base_idx = global_warp_idx * elements_per_warp;
    
    // Process multiple elements per warp
    #pragma unroll
    for (int elem = 0; elem < elements_per_warp; elem++) {
        const int idx = base_idx + elem;
        
        // Predicated execution instead of early return
        scalar_t sum = 0;
        if (idx < total_output) {
            const int outer_idx = idx / inner_size;
            const int inner_idx = idx % inner_size;
            const int64_t base_offset = outer_idx * reduce_size * inner_size + inner_idx;
            
            // Vectorized load when possible (assuming alignment)
            if (inner_size % 4 == 0 && lane % 4 == 0) {
                const scalar_t* input4 = input;
                for (int i = lane/4; i < reduce_size; i += warp_size/4) {
                    float4 val = input4[(base_offset + i * inner_size)/4];
                    sum += val.x + val.y + val.z + val.w;
                }
            } else {
                // Regular load path
                for (int i = lane; i < reduce_size; i += warp_size) {
                    sum += input[base_offset + i * inner_size];
                }
            }
            
            // Warp reduction using shuffle
            #pragma unroll
            for (int offset = warp_size/2; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            
            // Write result without branch
            if (lane == 0) {
                output[idx] = sum;
            }
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
    
    const int64_t total_output = outer_size * inner_size;
    const int threads_per_block = 128;
    const int elements_per_warp = 4;
    const int warps_per_block = threads_per_block / 32;
    const int elements_per_block = warps_per_block * elements_per_warp;
    const int blocks = (total_output + elements_per_block - 1) / elements_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        warp_sum_reduce_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size,
            total_output
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA)");
}