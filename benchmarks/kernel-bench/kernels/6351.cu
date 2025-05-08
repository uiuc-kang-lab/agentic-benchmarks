#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel performs sum reduction over a specified dimension using warp-level primitives with manual loop unrolling.
// Each warp computes one output element. The inner loop over the reduction dimension is manually unrolled
// to reduce loop overhead, and the warp-level reduction is performed using __shfl_down_sync.

template <typename scalar_t>
__global__ void manual_unroll_warp_reduce_sum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t total_outputs) {

    const int warpSize = 32;
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize;  // Unique warp ID
    int lane = global_thread_id % warpSize;      // Lane index within the warp
    int total_warps = (gridDim.x * blockDim.x) / warpSize;

    // Each warp processes one or more output elements in a grid-stride loop
    for (int out_idx = warp_id; out_idx < total_outputs; out_idx += total_warps) {
        // Map 1D output index to 2D indices: outer and inner
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;

        // Compute the base index for this output element
        int64_t base = outer_idx * reduce_size * inner_size + inner_idx;
        scalar_t sum = 0;

        // Calculate the number of iterations (each thread handles indices: lane, lane+warpSize, ...)
        int T = (reduce_size > lane) ? ((reduce_size - lane + warpSize - 1) / warpSize) : 0;
        
        // Manual unroll factor
        constexpr int unroll = 4;
        int unrolled_iters = (T / unroll) * unroll;

        // Manual unrolling: process multiple iterations per loop to reduce overhead
        #pragma unroll
        for (int j = 0; j < unrolled_iters; j += unroll) {
            int idx0 = lane + (j + 0) * warpSize;
            int idx1 = lane + (j + 1) * warpSize;
            int idx2 = lane + (j + 2) * warpSize;
            int idx3 = lane + (j + 3) * warpSize;
            sum += input[base + idx0 * inner_size];
            sum += input[base + idx1 * inner_size];
            sum += input[base + idx2 * inner_size];
            sum += input[base + idx3 * inner_size];
        }

        // Process any remaining iterations
        for (int j = unrolled_iters; j < T; j++) {
            int idx = lane + j * warpSize;
            sum += input[base + idx * inner_size];
        }

        // Warp-level reduction using shuffle down primitives (fully unrolled)
        unsigned int mask = 0xffffffff;
        sum += __shfl_down_sync(mask, sum, 16);
        sum += __shfl_down_sync(mask, sum, 8);
        sum += __shfl_down_sync(mask, sum, 4);
        sum += __shfl_down_sync(mask, sum, 2);
        sum += __shfl_down_sync(mask, sum, 1);

        // The first lane writes the result for this output element
        if (lane == 0) {
            output[out_idx] = sum;
        }
    }
}

// CUDA wrapper function
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    // Compute the product of dimensions before and after the reduction dimension
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Prepare the output tensor by setting the reduced dimension to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Total number of output elements equals outer_size * inner_size
    int64_t total_outputs = outer_size * inner_size;

    // Each output element is computed by one warp (32 threads)
    const int warpSize = 32;
    int total_threads = total_outputs * warpSize;
    int threads = 256;  // Must be a multiple of 32
    int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        manual_unroll_warp_reduce_sum_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA) with manual loop unrolling");
}
