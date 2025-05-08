#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define the segment size so that each block processes a contiguous, evenly distributed chunk of the vector
#define SEGMENT_SIZE 512

// Stage 1: Each block computes the sum-of-squares for its segment and atomically accumulates into the per-vector partial sum
template <typename scalar_t>
__global__ void l2norm_stage1_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ partial_sums,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int blocks_per_vector) {

    // Determine which vector and which segment this block is processing
    int vector_idx = blockIdx.x / blocks_per_vector;
    int block_segment = blockIdx.x % blocks_per_vector;
    if (vector_idx >= total_vectors) return;

    int base_offset = vector_idx * outer_stride;
    int seg_start = block_segment * SEGMENT_SIZE;
    int seg_end = seg_start + SEGMENT_SIZE;
    if (seg_end > C) seg_end = C;

    scalar_t sum = 0;
    // Each thread processes a subset of the segment via a stride-loop
    for (int i = seg_start + threadIdx.x; i < seg_end; i += blockDim.x) {
        scalar_t val = input[base_offset + i * stride_C];
        sum += val * val;
    }

    // Perform warp-level reduction using shuffle operations
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(__activemask(), sum, offset);
    }

    // Use shared memory to reduce across warps within the block
    __shared__ scalar_t sdata[32];  // enough for up to 32 warps
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces the values stored in shared memory
    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? sdata[lane] : 0;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(__activemask(), sum, offset);
        }
        if (threadIdx.x == 0) {
            // Atomically add the block's result to the global partial sum for this vector
            atomicAdd(&partial_sums[vector_idx], sum);
        }
    }
}

// Stage 2: Each block normalizes the corresponding segment using the computed norm
template <typename scalar_t>
__global__ void l2norm_stage2_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ partial_sums,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int blocks_per_vector) {

    int vector_idx = blockIdx.x / blocks_per_vector;
    int block_segment = blockIdx.x % blocks_per_vector;
    if (vector_idx >= total_vectors) return;

    int base_offset = vector_idx * outer_stride;
    int seg_start = block_segment * SEGMENT_SIZE;
    int seg_end = seg_start + SEGMENT_SIZE;
    if (seg_end > C) seg_end = C;

    // Compute normalization factor
    scalar_t norm = partial_sums[vector_idx];
    scalar_t inv_norm = 1.0 / (sqrt(norm) + 1e-12);

    // Each thread normalizes its assigned elements using a stride loop
    for (int i = seg_start + threadIdx.x; i < seg_end; i += blockDim.x) {
        output[base_offset + i * stride_C] = input[base_offset + i * stride_C] * inv_norm;
    }
}

// The forward function interfacing with PyTorch
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);
    auto partial_sums = torch::zeros({total_vectors}, input.options());

    // Calculate the number of blocks to evenly distribute each vector's workload
    int blocks_per_vector = (C + SEGMENT_SIZE - 1) / SEGMENT_SIZE;
    int total_blocks = total_vectors * blocks_per_vector;
    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_even_workload", ([&] {
        l2norm_stage1_kernel<scalar_t><<<total_blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            partial_sums.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride,
            blocks_per_vector
        );

        l2norm_stage2_kernel<scalar_t><<<total_blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            partial_sums.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride,
            blocks_per_vector
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization along dim=1 with even workload distribution");
}
