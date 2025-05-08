#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void coalesced_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (inner_idx >= inner_size) return;

    const int64_t input_offset = outer_idx * dim_size * inner_size + inner_idx;
    scalar_t max_val = input[input_offset];

    #pragma unroll 4
    for (int i = 1; i < dim_size; i++) {
        scalar_t val = input[input_offset + i * inner_size];
        max_val = max(max_val, val);
    }

    output[outer_idx * inner_size + inner_idx] = max_val;
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    int64_t dim_size = input.size(dim);

    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const int threads = 256;
    const int blocks_y = (inner_size + threads - 1) / threads;
    dim3 grid(outer_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        coalesced_max_reduce_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Max reduction forward (CUDA) with unrolled coalescing");
}
