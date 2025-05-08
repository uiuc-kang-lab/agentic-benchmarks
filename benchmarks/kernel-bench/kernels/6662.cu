#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void unrolled_coalesced_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = outer_size * inner_size;
    
    if (idx >= total_elements) return;
    
    const int outer_idx = idx / inner_size;
    const int inner_idx = idx % inner_size;
    
    const int64_t start = outer_idx * dim_size * inner_size + inner_idx;
    scalar_t max_val = input[start];

    #pragma unroll 4
    for (int i = 1; i < dim_size; ++i) {
        scalar_t val = input[start + i * inner_size];
        if (val > max_val) max_val = val;
    }
    
    output[idx] = max_val;
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++)
        outer_size *= input.size(i);
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++)
        inner_size *= input.size(i);
    
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        unrolled_coalesced_max_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            input.size(dim),
            inner_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA)");
}