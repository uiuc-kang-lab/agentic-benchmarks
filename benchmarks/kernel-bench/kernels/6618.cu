#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ scalar_t compute_max(const scalar_t* __restrict__ input, int64_t base_offset, int inner_idx, int64_t inner_size, int64_t dim_size) {
    scalar_t max_val = __ldg(&input[base_offset + inner_idx]);
    #pragma unroll 4
    for (int i = 1; i < dim_size; ++i) {
        scalar_t val = __ldg(&input[base_offset + i * inner_size + inner_idx]);
        max_val = max(max_val, val);
    }
    return max_val;
}

template <typename scalar_t>
__global__ void modular_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t inner_size,
    const int64_t dim_size
) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    if (inner_idx >= inner_size) return;

    int64_t base_offset = outer_idx * dim_size * inner_size;
    scalar_t max_val = compute_max(input, base_offset, inner_idx, inner_size, dim_size);
    output[outer_idx * inner_size + inner_idx] = max_val;
}

torch::Tensor modular_max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= input.size(i);
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); ++i) {
        inner_size *= input.size(i);
    }

    const int64_t dim_size = input.size(dim);

    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const int threads = 256;
    int blocks_y = (inner_size + threads - 1) / threads;
    dim3 grid(outer_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "modular_max_reduce_forward", ([&] {
        modular_max_reduce_kernel<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            inner_size,
            dim_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &modular_max_reduce_cuda_forward, "Modular Max reduce forward (CUDA)");
}