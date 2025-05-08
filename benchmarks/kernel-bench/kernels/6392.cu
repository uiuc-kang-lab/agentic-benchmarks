#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses warp-level primitives to perform reduction over the specified dimension.

template <typename scalar_t>
__global__ void warp_sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t total_output) {

    // Calculate global output index
    int idx = blockIdx.x * (blockDim.x / warpSize) + (threadIdx.x / warpSize);
    if (idx >= total_output) return;

    // Determine corresponding outer and inner indices
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;

    // Get lane ID within the warp
    const int lane = threadIdx.x & (warpSize - 1);
    // Get warp ID within the block
    const int warp_id = threadIdx.x / warpSize;

    scalar_t sum = 0;
    // Pre-calculate base offset to avoid redundant computations
    const int64_t base_offset = outer_idx * reduce_size * inner_size + inner_idx;
    
    // Each thread processes elements with stride of warpSize
    #pragma unroll 4
    for (int i = lane; i < reduce_size; i += warpSize) {
        int64_t offset = base_offset + i * inner_size;
        sum += input[offset];
    }

    // Warp-level reduction using shuffle
    const unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // First thread in each warp writes the result
    if (lane == 0) {
        output[idx] = sum;
    }
}

// Host function wrapping the kernel launch

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Adjust negative dimensions
    if (dim < 0) dim += input.dim();
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    // Compute outer_size: product of dimensions before the reduction dim
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    // Compute inner_size: product of dimensions after the reduction dim
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Prepare output tensor with reduce dimension set to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Total number of output elements
    int64_t total_output = outer_size * inner_size;

    // Launch one warp (32 threads) per output element
    const int threads = 32;  // warp size
    const int blocks = total_output;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        warp_sum_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size,
            total_output
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA) using warp-level primitives");
}
