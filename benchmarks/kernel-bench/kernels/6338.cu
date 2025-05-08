#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Modular device function: warp-level reduction
template <typename scalar_t>
__device__ inline scalar_t warpReduceSum(scalar_t val) {
    const int warpSize = 32;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Modular device function: compute the sum for a single output element
template <typename scalar_t>
__device__ inline void compute_output_sum(const scalar_t* __restrict__ input,
                                           scalar_t* __restrict__ output,
                                           int64_t outer_idx,
                                           int64_t inner_idx,
                                           int64_t reduce_size,
                                           int64_t inner_size) {
    const int warpSize = 32;
    int lane = threadIdx.x % warpSize;
    int64_t base = outer_idx * reduce_size * inner_size + inner_idx;
    scalar_t sum_val = 0;
    // Each thread in the warp processes part of the reduction dimension
    for (int i = lane; i < reduce_size; i += warpSize) {
        sum_val += input[base + i * inner_size];
    }
    // Reduce within the warp using the modular function
    sum_val = warpReduceSum(sum_val);
    // First lane writes back the result
    if (lane == 0) {
        output[outer_idx * inner_size + inner_idx] = sum_val;
    }
}

// Kernel that assigns each warp to compute one output element in a grid-stride loop
template <typename scalar_t>
__global__ void modular_warp_reduce_sum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t total_outputs) {

    const int warpSize = 32;
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize;
    int total_warps = (gridDim.x * blockDim.x) / warpSize;

    // Each warp processes one or more output elements via grid-stride loop
    for (int out_idx = warp_id; out_idx < total_outputs; out_idx += total_warps) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        compute_output_sum(input, output, outer_idx, inner_idx, reduce_size, inner_size);
    }
}

// CUDA wrapper function
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Adjust negative dimensions
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

    // Set the reduced dimension to 1 for the output tensor
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Total number of output elements
    int64_t total_outputs = outer_size * inner_size;

    // Configure kernel launch parameters: one warp (32 threads) per output element
    const int warpSize = 32;
    int required_warps = total_outputs; // one warp per output element
    int total_threads = required_warps * warpSize;
    int threads = 256; // multiple of 32 for efficiency
    int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        modular_warp_reduce_sum_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &sum_reduce_cuda, "Modular warp-level sum reduction forward (CUDA)");
}
