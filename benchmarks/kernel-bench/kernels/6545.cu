#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void coalesced_mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {
    
    const int outer = blockIdx.x; // Cache outer index for reuse
    const int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (outer >= outer_size || inner_idx >= inner_size) return;
    
    scalar_t sum = 0.0;
    const int input_base = outer * dim_size * inner_size + inner_idx;
    
    #pragma unroll 4
    for (int i = 0; i < dim_size; ++i) {
        sum += input[input_base + i * inner_size];
    }
    
    output[outer * inner_size + inner_idx] = sum / static_cast<scalar_t>(dim_size);
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
    
    const int threads_per_block = 256;
    const dim3 grid(outer_size, (inner_size + threads_per_block - 1) / threads_per_block);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "coalesced_mean_reduce_cuda", ([&] {
        coalesced_mean_reduce_kernel<scalar_t><<<grid, threads_per_block>>>(
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
    m.def("forward", &mean_reduce_cuda, "Coalesced mean reduction (CUDA)");
}