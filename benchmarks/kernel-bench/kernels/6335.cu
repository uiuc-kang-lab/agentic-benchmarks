#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses warp-level primitives and experiments with block sizes.
// Each warp is assigned to compute one output element via reduction along the specified dimension.

template <typename scalar_t>
__global__ void warp_reduce_sum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t total_outputs) {

    const int warpSize = 32;
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize;
    int lane = global_thread_id % warpSize;

    // Total number of warps launched
    int total_warps = (gridDim.x * blockDim.x) / warpSize;

    // Each warp processes output elements in a grid-stride loop
    for (int out_idx = warp_id; out_idx < total_outputs; out_idx += total_warps) {
        // Map the 1D output index to outer and inner indices
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;

        int64_t base = outer_idx * reduce_size * inner_size + inner_idx;
        scalar_t sum_val = 0;

        // Each thread in the warp sums over part of the reduction dimension
        for (int i = lane; i < reduce_size; i += warpSize) {
            sum_val += input[base + i * inner_size];
        }

        // Perform warp-level reduction using shuffle down
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_val += __shfl_down_sync(0xFFFFFFFF, sum_val, offset);
        }

        // First lane writes the final result
        if (lane == 0) {
            output[out_idx] = sum_val;
        }
    }
}

// CUDA wrapper function
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Adjust for negative dimensions
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

    // The output tensor has the reduction dimension set to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Total number of output elements
    int64_t total_outputs = outer_size * inner_size;

    // Each output element is computed by a warp (32 threads), so total threads needed:
    const int warpSize = 32;
    int total_threads = total_outputs * warpSize;

    // Experiment with block sizes from candidate list {32, 64, 128, 256, 512}
    int block_size = 32;
    if (total_threads >= 512) {
        block_size = 512;
    } else if (total_threads >= 256) {
        block_size = 256;
    } else if (total_threads >= 128) {
        block_size = 128;
    } else if (total_threads >= 64) {
        block_size = 64;
    } else {
        block_size = 32;
    }

    // Determine number of blocks
    int blocks = (total_threads + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        warp_reduce_sum_kernel<scalar_t><<<blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size,
            total_outputs
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA)");
}
