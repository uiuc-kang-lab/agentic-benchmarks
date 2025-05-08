#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using manually unrolled loops with warp-level primitives for reduction.
// Each warp handles one output, fully unrolling the loop over the reduction dimension to minimize overhead.

template <typename scalar_t>
__global__ void unrolled_sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t total_outputs) {
    
    const int warpSize = 32;
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize;
    int lane = global_thread_id % warpSize;

    int total_warps = (gridDim.x * blockDim.x) / warpSize;

    for (int out_idx = warp_id; out_idx < total_outputs; out_idx += total_warps) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;

        int64_t base = outer_idx * reduce_size * inner_size + inner_idx;
        scalar_t sum_val = 0;

        // Manually unroll reduction with warp shuffle down
        for (int i = 0; i < reduce_size; i += warpSize) {
            #pragma unroll
            for (int j = 0; j < warpSize; ++j) {
                if (i + j < reduce_size) {
                    sum_val += input[base + (i + j) * inner_size];
                }
            }
        }

        // Warp reduction using shuffle
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_val += __shfl_down_sync(0xFFFFFFFF, sum_val, offset);
        }

        if (lane == 0) {
            output[out_idx] = sum_val;
        }
    }
}

// CUDA wrapper to perform sum reduction
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

    int64_t total_outputs = outer_size * inner_size;

    const int warpSize = 32;
    int total_threads = total_outputs * warpSize;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        unrolled_sum_reduce_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA) with unrolled loops");
}