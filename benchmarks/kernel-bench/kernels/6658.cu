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
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (inner_idx >= inner_size) return;

    const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
    scalar_t max_val = __ldg(input + base_offset);

    int i = 1;
    #pragma unroll 8
    for (; i <= dim_size - 8; i += 8) {
        scalar_t v1 = __ldg(input + base_offset + i*inner_size);
        scalar_t v2 = __ldg(input + base_offset + (i+1)*inner_size);
        scalar_t v3 = __ldg(input + base_offset + (i+2)*inner_size);
        scalar_t v4 = __ldg(input + base_offset + (i+3)*inner_size);
        scalar_t v5 = __ldg(input + base_offset + (i+4)*inner_size);
        scalar_t v6 = __ldg(input + base_offset + (i+5)*inner_size);
        scalar_t v7 = __ldg(input + base_offset + (i+6)*inner_size);
        scalar_t v8 = __ldg(input + base_offset + (i+7)*inner_size);
        scalar_t max1 = max(max(max(v1, v2), max(v3, v4)), max(max(v5, v6), max(v7, v8)));
        max_val = max(max_val, max1);
    }
    
    for (; i < dim_size; i++) {
        max_val = max(max_val, __ldg(input + base_offset + i*inner_size));
    }

    output[outer_idx * inner_size + inner_idx] = max_val;
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= input.size(i);

    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) inner_size *= input.size(i);

    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const int threads = 256;
    const int blocks_y = (inner_size + threads - 1) / threads;
    dim3 grid(outer_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        unrolled_coalesced_max_reduce_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Unrolled coalesced max reduction (CUDA)");
}