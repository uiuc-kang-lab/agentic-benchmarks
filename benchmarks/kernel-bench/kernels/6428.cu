#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// This kernel uses cooperative groups to partition threads into a warp and performs a uniform, divergence-free reduction
// using the __shfl_xor_sync intrinsic. The warp leader (determined uniformly via cooperative groups) writes the final sum.

template <typename scalar_t>
__global__ void cg_warp_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size) {

    constexpr int WARP_SIZE = 32;
    const unsigned int FULL_MASK = 0xffffffff;

    // Partition the block into a warp of 32 threads with cooperative groups for uniform control flow
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

    // Each block computes one output element (flattened index over outer * inner dimensions)
    int out_idx = blockIdx.x;
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;
    int64_t base = outer_idx * reduce_size * inner_size + inner_idx;

    scalar_t sum = 0;
    // Each thread aggregates a portion of the reduction dimension in a uniform loop
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        sum += input[base + i * inner_size];
    }

    // Perform warp-level reduction using shuffle XOR, which avoids divergent branches
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(FULL_MASK, sum, offset);
    }

    // The warp leader (uniformly determined) writes the result
    if (warp.thread_rank() == 0) {
        output[out_idx] = sum;
    }
}

// Host wrapper to set up dimensions and launch the kernel

torch::Tensor cg_warp_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
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

    // The reduction dimension is collapsed to 1 in the output
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Total number of output elements
    int64_t num_outputs = outer_size * inner_size;
    // Launch one block per output element, with one warp (32 threads) per block
    int threads = 32;
    int blocks = num_outputs;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cg_warp_reduce_cuda", ([&] {
        cg_warp_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cg_warp_reduce_cuda, "Sum reduction with cooperative groups and uniform control flow (CUDA)");
}
