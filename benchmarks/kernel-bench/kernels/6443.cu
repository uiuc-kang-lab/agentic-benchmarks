#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


// Warp-level reduction using __shfl_down_sync. This function reduces the value across the warp.
template <typename scalar_t>
__inline__ __device__ scalar_t warp_reduce_sum(scalar_t val) {
    // Full mask for active threads
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Each warp cooperatively computes the mean reduction over the reduction dimension
// Each warp handles one output element. Threads in the warp load different elements and then reduce using warp intrinsics.

template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

    // Each warp computes one output element.
    int warpsPerBlock = blockDim.x / warpSize;
    int global_warp_id = blockIdx.x * warpsPerBlock + threadIdx.x / warpSize;

    // Total number of output elements
    int64_t total_outputs = outer_size * inner_size;
    if (global_warp_id >= total_outputs) return;

    // Compute the corresponding outer and inner indices
    int outer_idx = global_warp_id / inner_size;
    int inner_idx = global_warp_id % inner_size;

    // Starting pointer for the reduction for this output element
    // Input layout: [outer_size, dim_size, inner_size]
    int64_t base = outer_idx * dim_size * inner_size + inner_idx;

    unsigned int lane = threadIdx.x & (warpSize - 1);
    scalar_t partial_sum = 0;

    // Each thread in the warp loads a subset of elements along the reduction dimension
    for (int i = lane; i < dim_size; i += warpSize) {
        // Compute index: for given i in dim_size, the offset in memory is base + i * inner_size
        partial_sum += __ldg(input + base + i * inner_size);
    }

    // Perform warp-level reduction
    partial_sum = warp_reduce_sum(partial_sum);

    // The first lane in each warp writes the result
    if (lane == 0) {
        output[global_warp_id] = partial_sum / static_cast<scalar_t>(dim_size);
    }
}


// CUDA kernel launcher
// Computes mean reduction over the given dimension using warp-level primitives to reduce synchronization overhead.

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimension
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    // Calculate outer_size and inner_size
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Prepare output tensor: remove the reduction dimension
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    // Total number of output elements equals outer_size * inner_size
    int64_t total_outputs = outer_size * inner_size;

    // Set block size as a multiple of warpSize (e.g., 256 threads per block, which gives 8 warps per block)
    int threads = 256;
    int warpsPerBlock = threads / warpSize;  // typically 8 warps per block
    int blocks = (total_outputs + warpsPerBlock - 1) / warpsPerBlock;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &mean_reduce_cuda, "Mean reduction (CUDA) using warp-level primitives");
}
