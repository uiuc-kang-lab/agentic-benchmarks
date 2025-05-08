#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: Compute partial sum of squares for each vector using multiple blocks per vector.
// Each block computes a partial sum over a segment of the vector and adds it atomically to a global sum array.

template <typename scalar_t>
__global__ void l2_norm_partial_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ global_sum,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int blocks_per_vector) {

    // Determine which vector and segment this block is responsible for
    int vector_idx = blockIdx.x / blocks_per_vector;
    int seg_idx = blockIdx.x % blocks_per_vector;
    if (vector_idx >= total_vectors) return;

    // Compute segment boundaries for the current vector
    int segment_length = (C + blocks_per_vector - 1) / blocks_per_vector;  // ceil division
    int start = seg_idx * segment_length;
    int end = start + segment_length;
    if (end > C) end = C;

    int base_offset = vector_idx * outer_stride;

    // Each thread computes a partial sum over its assigned indices in the segment
    scalar_t partial = 0;
    for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
        scalar_t val = input[base_offset + k * stride_C];
        partial += val * val;
    }

    // Reduce partial sums within the block using shared memory
    __shared__ scalar_t sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = partial;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread in the block atomically adds the block's sum to the global sum
    if (tid == 0) {
        atomicAdd(&global_sum[vector_idx], sdata[0]);
    }
}

// Kernel 2: Normalize each vector using the computed L2 norm.
// The grid is organized similarly to kernel 1 to cover all elements of each vector.

template <typename scalar_t>
__global__ void l2_normalize_kernel_phase2(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ global_sum,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int blocks_per_vector_norm) {

    int vector_idx = blockIdx.x / blocks_per_vector_norm;
    int seg_idx = blockIdx.x % blocks_per_vector_norm;
    if (vector_idx >= total_vectors) return;

    int segment_length = (C + blocks_per_vector_norm - 1) / blocks_per_vector_norm;
    int start = seg_idx * segment_length;
    int end = start + segment_length;
    if (end > C) end = C;

    int base_offset = vector_idx * outer_stride;

    // Each block computes the normalization factor (redundantly, but cheaply) from the global sum
    scalar_t norm = sqrt(global_sum[vector_idx]) + 1e-12;
    scalar_t inv_norm = (scalar_t)1.0 / norm;

    // Normalize the elements in the segment
    for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
        output[base_offset + k * stride_C] = input[base_offset + k * stride_C] * inv_norm;
    }
}


// The forward function orchestrates the two kernels to perform L2 normalization along dim=1.

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    // Assume input shape is [N, C] or contiguous equivalent
    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);
    auto global_sum = torch::zeros({total_vectors}, input.options());

    // Decide on the number of blocks per vector based on a segment size (e.g., 1024 elements per block)
    int seg_size = 1024;
    int blocks_per_vector = (C + seg_size - 1) / seg_size;
    if (blocks_per_vector < 1) blocks_per_vector = 1;
    int total_blocks = total_vectors * blocks_per_vector;

    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_partial", ([&] {
        l2_norm_partial_kernel<scalar_t><<<total_blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            global_sum.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride,
            blocks_per_vector
        );
    }));
    
    // Launch normalization kernel with the same grid configuration for simplicity
    int total_blocks_norm = total_vectors * blocks_per_vector;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize_phase2", ([&] {
        l2_normalize_kernel_phase2<scalar_t><<<total_blocks_norm, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            global_sum.data_ptr<scalar_t>(),
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
    m.def("forward", &forward, "L2 normalization along dim=1 with atomic operations only where necessary");
}
