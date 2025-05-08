#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define the segment size to determine the workload per block per vector segment
#define SEGMENT_SIZE 1024

// Stage 1: Compute the partial squared sum for each vector segment using stride loops
template <typename scalar_t>
__global__ void l2_normalize_stage1_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ partial_sums,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int blocks_per_vector) {

    // Determine which vector and segment this block is responsible for
    int vector_idx = blockIdx.x / blocks_per_vector;
    int seg_idx = blockIdx.x % blocks_per_vector;
    if (vector_idx >= total_vectors) return;

    int base_offset = vector_idx * outer_stride;
    int seg_start = seg_idx * SEGMENT_SIZE;
    int seg_end = seg_start + SEGMENT_SIZE;
    if (seg_end > C) seg_end = C;

    scalar_t sum = 0;

    // Use a stride loop to cover all elements in the segment
    // Each thread processes multiple elements spaced by blockDim.x
    for (int i = seg_start + threadIdx.x; i < seg_end; i += blockDim.x) { // Ensure coalesced memory access
        scalar_t val = input[base_offset + i * stride_C];
        sum += val * val;
    }

    // Warp-level reduction using shuffle operations
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Use shared memory to reduce across warps in this block
    __shared__ scalar_t shared_mem[32]; // Enough to hold one value per warp
    int warp_id = threadIdx.x / warpSize;
    if ((threadIdx.x % warpSize) == 0) {
        shared_mem[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces the values in shared memory
    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_mem[threadIdx.x] : 0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) {
            // Atomic addition to accumulate the partial sum for the vector
            atomicAdd(&partial_sums[vector_idx], sum);
        }
    }
}

// Stage 2: Normalize the vector using the computed partial sum
// Each block covers a segment of the vector and applies the normalization using stride loops.
template <typename scalar_t>
__global__ void l2_normalize_stage2_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ partial_sums,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int blocks_per_vector) {

    int vector_idx = blockIdx.x / blocks_per_vector;
    int seg_idx = blockIdx.x % blocks_per_vector;
    if (vector_idx >= total_vectors) return;

    int base_offset = vector_idx * outer_stride;
    int seg_start = seg_idx * SEGMENT_SIZE;
    int seg_end = seg_start + SEGMENT_SIZE;
    if (seg_end > C) seg_end = C;

    // Compute the normalization factor from the accumulated squared sum
    scalar_t norm = partial_sums[vector_idx];
    scalar_t inv_norm = 1.0 / (sqrt(norm) + 1e-12);

    // Use a stride loop to normalize each element in the segment
    for (int i = seg_start + threadIdx.x; i < seg_end; i += blockDim.x) { // Ensure coalesced memory access
        output[base_offset + i * stride_C] = input[base_offset + i * stride_C] * inv_norm;
    }
}

// The forward function prepares the kernel execution and launches the two stages
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "Need at least 1 dimension");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);
    auto partial_sums = torch::zeros({total_vectors}, input.options());

    const int threads = 256;
    // Calculate the number of blocks per vector based on the segment size
    const int blocks_per_vector = (C + SEGMENT_SIZE - 1) / SEGMENT_SIZE;
    const int total_blocks = total_vectors * blocks_per_vector;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize_stride", ([&] {
        l2_normalize_stage1_kernel<scalar_t><<<total_blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            partial_sums.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride,
            blocks_per_vector
        );

        l2_normalize_stage2_kernel<scalar_t><<<total_blocks, threads>>>(
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
    m.def("forward", &forward, "L2 normalization along dim=1 using stride loops");
}
