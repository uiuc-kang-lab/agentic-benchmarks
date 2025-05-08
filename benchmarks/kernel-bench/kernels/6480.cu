#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ scalar_t warp_reduce_sum(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t outer_size,
    int64_t dim_size, 
    int64_t inner_size) {

    const int warpSize = 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane = tid % warpSize;
    if (warp_id >= outer_size * inner_size) return;

    const int outer_idx = warp_id / inner_size;
    const int inner_idx = warp_id % inner_size;
    const int input_offset = outer_idx * dim_size * inner_size + inner_idx;

    scalar_t sum = 0;
    // Each lane processes a subset of the reduction dimension
    for (int i = lane; i < dim_size; i += warpSize) {
        sum += input[input_offset + i * inner_size];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        output[warp_id] = sum / dim_size;
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
    
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &mean_reduce_cuda, "Mean reduction (CUDA)");
}