#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel performs sum reduction over a given dimension using warp-level primitives with full unrolling.
// Each warp is responsible for computing one output element by summing over a segment of the reduction dimension.
// The reduction within each warp is fully unrolled using __shfl_down_sync, eliminating the use of shared memory and reducing overhead.

template <typename scalar_t>
__global__ void fully_unrolled_warp_reduce_sum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t total_outputs) {

    const int warpSize = 32;
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize;    // Which warp this thread belongs to
    int lane = global_thread_id % warpSize;         // Lane index within the warp
    int total_warps = (gridDim.x * blockDim.x) / warpSize;

    // Each warp computes one output element, with a grid-stride loop over outputs
    for (int out_idx = warp_id; out_idx < total_outputs; out_idx += total_warps) {
        // Map the 1D output index to (outer, inner) indices
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;

        // Base index for the given outer and inner indices
        int64_t base = outer_idx * reduce_size * inner_size + inner_idx;
        scalar_t sum = 0;

        // Each thread accumulates a partial sum from its portion of the reduction dimension
        for (int i = lane; i < reduce_size; i += warpSize) {
            sum += input[base + i * inner_size];
        }

        // Fully unroll warp-level reduction using shuffle down
        unsigned int mask = 0xffffffff;
        sum += __shfl_down_sync(mask, sum, 16);
        sum += __shfl_down_sync(mask, sum, 8);
        sum += __shfl_down_sync(mask, sum, 4);
        sum += __shfl_down_sync(mask, sum, 2);
        sum += __shfl_down_sync(mask, sum, 1);

        // The first lane writes the final result
        if (lane == 0) {
            output[out_idx] = sum;
        }
    }
}

// CUDA wrapper function
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Adjust negative dimensions
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    // Compute outer and inner sizes
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Set the reduced dimension to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Total number of outputs is outer_size * inner_size
    int64_t total_outputs = outer_size * inner_size;

    // Each output element is computed by one warp (32 threads)
    const int warpSize = 32;
    int total_threads = total_outputs * warpSize;
    int threads = 256;  // Must be a multiple of 32
    int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        fully_unrolled_warp_reduce_sum_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA) using fully unrolled warp-level primitives");
}
