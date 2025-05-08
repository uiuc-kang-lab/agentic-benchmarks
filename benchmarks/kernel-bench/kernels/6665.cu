#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void warp_reduce_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    const int outer_idx = blockIdx.x;
    const int warp_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int inner_idx = warp_id;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    const int lane_id = threadIdx.x;
    const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

    scalar_t max_val = -INFINITY;
    const int elements_per_thread = (dim_size + 31) / 32;

    for (int k = 0; k < elements_per_thread; ++k) {
        int i = lane_id + k * 32;
        if (i < dim_size) {
            scalar_t val = input[base_offset + i * inner_size];
            max_val = max(max_val, val);
        }
    }

    // Warp reduction with butterfly shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        scalar_t tmp = __shfl_down_sync(__activemask(), max_val, offset);
        max_val = max(max_val, tmp);
    }

    if (lane_id == 0) {
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

    const int warps_per_block = 4;
    dim3 block(32, warps_per_block);
    dim3 grid(outer_size, (inner_size + warps_per_block - 1) / warps_per_block);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        warp_reduce_max_reduce_kernel<scalar_t><<<grid, block>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Warp Reduce Max Reduction Forward (CUDA)");
}