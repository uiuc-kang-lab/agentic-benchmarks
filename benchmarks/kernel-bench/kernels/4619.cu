#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Define block dimensions for optimized indexing
#define OFFSETS_PER_BLOCK 32
#define THREADS_FEATURE 8

// Optimized kernel with improved thread and block indexing

template <typename scalar_t>
__global__ void rms_norm_optimized_indexing_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int total_offsets,  // batch_size * numel_per_batch
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    // Calculate the global offset index corresponding to a (batch, offset) pair
    int global_offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_offset >= total_offsets) return;

    // Determine the batch id and the offset within the batch
    int batch_id = global_offset / numel_per_batch;
    int offset = global_offset % numel_per_batch;
    int base = batch_id * num_features * numel_per_batch;

    // Shared memory for reduction: size = OFFSETS_PER_BLOCK * THREADS_FEATURE
    __shared__ scalar_t sdata[OFFSETS_PER_BLOCK * THREADS_FEATURE];

    // Each thread in the column computes a partial sum over a subset of feature indices
    scalar_t partial_sum = 0;
    for (int f = threadIdx.y; f < num_features; f += THREADS_FEATURE) {
        int pos = base + f * numel_per_batch + offset;
        scalar_t val = input[pos];
        partial_sum += val * val;
    }

    // Store the partial sum in shared memory. Shared memory is laid out as [THREADS_FEATURE][OFFSETS_PER_BLOCK]
    int smem_index = threadIdx.y * OFFSETS_PER_BLOCK + threadIdx.x;
    sdata[smem_index] = partial_sum;
    __syncthreads();

    // Perform reduction along the feature dimension (vertical reduction within the column)
    for (int stride = THREADS_FEATURE / 2; stride > 0; stride /= 2) {
        if (threadIdx.y < stride) {
            sdata[smem_index] += sdata[(threadIdx.y + stride) * OFFSETS_PER_BLOCK + threadIdx.x];
        }
        __syncthreads();
    }

    // Thread with threadIdx.y == 0 in each column now holds the complete sum of squares
    scalar_t rms;
    if (threadIdx.y == 0) {
        scalar_t sumsq = sdata[threadIdx.x];
        rms = sqrt(sumsq / num_features + eps);
        // Store the computed rms in shared memory for use by all threads in this column
        sdata[threadIdx.x] = rms;
    }
    __syncthreads();
    rms = sdata[threadIdx.x];

    // Normalization: each thread in the column normalizes a subset of feature elements
    for (int f = threadIdx.y; f < num_features; f += THREADS_FEATURE) {
        int pos = base + f * numel_per_batch + offset;
        scalar_t val = input[pos];
        output[pos] = val / rms;
    }
}

// CUDA forward function with optimized indexing

torch::Tensor rms_norm_cuda_forward_optimized_indexing(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Total number of (batch, offset) pairs to process
    int total_offsets = batch_size * numel_per_batch;

    // Define block dimensions: each block has OFFSETS_PER_BLOCK columns and THREADS_FEATURE rows
    dim3 block(OFFSETS_PER_BLOCK, THREADS_FEATURE);
    int grid_x = (total_offsets + OFFSETS_PER_BLOCK - 1) / OFFSETS_PER_BLOCK;
    dim3 grid(grid_x);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_optimized_indexing", ([&] {
        rms_norm_optimized_indexing_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_offsets,
            num_features,
            numel_per_batch,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward_optimized_indexing, "RMS normalization forward with optimized indexing (CUDA)");
}
