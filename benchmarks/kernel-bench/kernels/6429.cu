#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void grid_strided_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t outer_size,
    int64_t inner_size) {

    const int tid = threadIdx.x;
    const int bid_x = blockIdx.x;  // for inner dimension
    const int bid_y = blockIdx.y;  // for outer dimension
    const unsigned int FULL_MASK = 0xffffffff;
    
    for (int outer_idx = bid_y; outer_idx < outer_size; outer_idx += gridDim.y) {
        for (int inner_idx = bid_x; inner_idx < inner_size; inner_idx += gridDim.x) {
            const int out_idx = outer_idx * inner_size + inner_idx;
            const int64_t base = outer_idx * reduce_size * inner_size + inner_idx;
            
            scalar_t sum = 0;
            #pragma unroll 4
            for (int i = tid; i < reduce_size; i += blockDim.x) {
                sum += input[base + i * inner_size];
            }
            
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(FULL_MASK, sum, offset);
            }
            
            if (tid == 0) {
                output[out_idx] = sum;
            }
        }
    }
}

torch::Tensor grid_strided_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= sizes[i];
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) inner_size *= sizes[i];
    
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());
    
    const int threads = std::min((int64_t)256, reduce_size);
    dim3 blocks;
    blocks.x = std::min(inner_size, static_cast<int64_t>(256));
    blocks.y = std::min(outer_size, static_cast<int64_t>(256));
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_strided_reduce_cuda", ([&] {
        grid_strided_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            outer_size,
            inner_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &grid_strided_reduce_cuda, "Grid-strided reduction (CUDA)");
}