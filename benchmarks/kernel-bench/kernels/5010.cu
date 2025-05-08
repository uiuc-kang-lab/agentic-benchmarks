/*
This CUDA code implements L2 normalization along dim=1 in one cooperative kernel launch.
It combines the segmented reduction (for large C) and normalization phases into a
single kernel call using CUDA cooperative groups to perform grid synchronization.
Note: This kernel must be launched as a cooperative kernel (using cudaLaunchCooperativeKernel)
and requires hardware and driver support for grid-wide synchronization.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Cooperative kernel that performs segmented reduction and normalization in one launch

template <typename scalar_t>
__global__ void l2_normalize_coop_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ global_sums,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int elements_per_block) {

    // Determine segmentation parameters
    const int blocks_per_vector = (C + elements_per_block - 1) / elements_per_block;
    const int vector_idx = blockIdx.x / blocks_per_vector;
    const int segment_idx = blockIdx.x % blocks_per_vector;
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * outer_stride;
    const int segment_start = segment_idx * elements_per_block;
    int segment_end = segment_start + elements_per_block;
    if (segment_end > C) segment_end = C;

    // Each block computes the sum of squares for its assigned segment
    scalar_t local_sum = 0;
    for (int i = segment_start + threadIdx.x; i < segment_end; i += blockDim.x) {
        scalar_t val = input[base_offset + i * stride_C];
        local_sum += val * val;
    }

    // Intra-warp reduction using shuffle
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Use shared memory to accumulate warp-level results
    __shared__ scalar_t shared[32];  // Enough to hold results for up to 32 warps
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared[warp_id] = local_sum;
    }
    __syncthreads();

    // First thread reduces across warps in the block
    scalar_t block_sum = 0;
    const int num_warps = (blockDim.x + 31) / 32;
    if (threadIdx.x < num_warps) {
        block_sum = shared[threadIdx.x];
        for (int offset = num_warps/2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
    }

    if (threadIdx.x == 0) {
        // Atomic add the block's partial sum to the global sum for this vector
        atomicAdd(&global_sums[vector_idx], block_sum);
    }

    // Synchronize all blocks in the grid using cooperative groups
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    // Compute the normalization factor (1/sqrt(sum)) once per vector
    scalar_t norm = 1.0;
    if (threadIdx.x == 0) {
        norm = 1.0 / (sqrt(global_sums[vector_idx]) + static_cast<scalar_t>(1e-12));
    }
    // Broadcast norm within the warp
    norm = __shfl_sync(0xffffffff, norm, 0);

    // Each block normalizes its assigned segment
    for (int i = segment_start + threadIdx.x; i < segment_end; i += blockDim.x) {
        scalar_t val = input[base_offset + i * stride_C];
        output[base_offset + i * stride_C] = val * norm;
    }
}


// Host forward function

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0); // Assumes contiguous layout across the first dimension

    auto output = torch::empty_like(input);
    // Allocate global memory for accumulating partial sums per vector
    auto global_sums = torch::zeros({total_vectors}, input.options());

    const int threads = 256;
    // Parameter: number of elements to process per block per vector segment
    const int elements_per_block = 1024;

    // Determine the number of blocks per vector
    const int blocks_per_vector = (C + elements_per_block - 1) / elements_per_block;
    const int total_blocks = blocks_per_vector * total_vectors;

    // Launch the cooperative kernel. Make sure to use cudaLaunchCooperativeKernel from the host if needed.
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize_coop", ([&] {
        void* args[] = {
            (void*)&(input.data_ptr<scalar_t>()),
            (void*)&(output.data_ptr<scalar_t>()),
            (void*)&(global_sums.data_ptr<scalar_t>()),
            (void*)&C,
            (void*)&total_vectors,
            (void*)&stride_C,
            (void*)&outer_stride,
            (void*)&elements_per_block
        };
        // Launch using the cooperative kernel API
        cudaLaunchCooperativeKernel(
            (void*)l2_normalize_coop_kernel<scalar_t>,
            total_blocks,
            threads,
            args
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization along dim=1 using a cooperative kernel");
}
