#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void coalesced_sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size) {

    const int warpSize = 32;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;

    int outer_idx = warp_id / ((inner_size + warpSize - 1) / warpSize);
    int inner_block = warp_id % ((inner_size + warpSize - 1) / warpSize);
    int inner_start = inner_block * warpSize;

    if (outer_idx >= outer_size) return;

    scalar_t sum = 0;
    int inner_idx = inner_start + lane;
    if (inner_idx < inner_size) {
        for (int r = 0; r < reduce_size; ++r) {
            int64_t input_idx = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
            sum += input[input_idx];
        }
        output[outer_idx * inner_size + inner_idx] = sum;
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

    const int warpSize = 32;
    int total_warps = outer_size * ((inner_size + warpSize - 1) / warpSize);
    int total_threads = total_warps * warpSize;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        coalesced_sum_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Coalesced sum reduction forward (CUDA)");
}