#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using warp-level primitives with loop unrolling for reduction across the reduce dimension.
// Each warp computes one output element by partitioning the reduction workload among its threads.

template <typename scalar_t>
__global__ void unroll_warp_reduce_sum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t total_outputs) {

    // Calculate global warp id and lane id
    const int warpSize = 32;
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize;
    int lane = global_thread_id % warpSize;

    // Total warps available
    int total_warps = (gridDim.x * blockDim.x) / warpSize;

    // Each warp processes one output element in a grid-stride loop over warps
    for (int out_idx = warp_id; out_idx < total_outputs; out_idx += total_warps) {
        // Map the output index to the corresponding outer and inner indices
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;

        // Compute the base address for the reduction
        int64_t base = outer_idx * reduce_size * inner_size + inner_idx;
        scalar_t sum_val = 0;

        // Each thread in the warp accumulates a partial sum over the reduction dimension, striding by warpSize
        #pragma unroll
        for (int i = lane; i < reduce_size; i += warpSize) {
            sum_val += input[base + i * inner_size];
        }

        // Perform warp-level reduction using shuffle down
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_val += __shfl_down_sync(0xFFFFFFFF, sum_val, offset);
        }

        // Lane 0 writes the final result for this output element
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

    // Compute outer and inner dimensions
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Output tensor: replacing reduction dimension with 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Total number of output elements is outer_size x inner_size
    int64_t total_outputs = outer_size * inner_size;

    // Configure kernel launch parameters using warp-level reduction
    // Each output element is computed by one warp (32 threads)
    const int warpSize = 32;
    int required_warps = total_outputs;      // one warp per output element
    int total_threads = required_warps * warpSize;
    int threads = 256;                       // Choose block size as a multiple of 32 (e.g., 256 threads per block)
    int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        unroll_warp_reduce_sum_kernel<scalar_t><<<blocks, threads>>>(
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
