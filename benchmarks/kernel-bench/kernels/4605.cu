#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// This kernel uses a 2D grid where:
//   - gridDim.y corresponds to the batch dimension
//   - gridDim.x and threadIdx.x correspond to the spatial (numel_per_batch) dimension
// This ensures that threads in a warp (with consecutive threadIdx.x values) access consecutive memory locations within a feature, thus achieving coalesced global memory accesses.

template <typename scalar_t>
__global__ void rms_norm_2d_coalesced_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    // Get batch index from the 2D grid
    const int batch_id = blockIdx.y;
    // Get spatial offset index within the batch
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds
    if (offset >= numel_per_batch) return;

    // Compute the starting index for this batch
    const int batch_offset = batch_id * num_features * numel_per_batch;

    // Compute sum of squares for current spatial location across all features
    scalar_t sumsq = static_cast<scalar_t>(0);
    for (int feat = 0; feat < num_features; feat++) {
        int idx = batch_offset + feat * numel_per_batch + offset;
        scalar_t val = input[idx];
        sumsq += val * val;
    }

    // Compute RMS value with epsilon for numerical stability
    scalar_t rms = sqrt(sumsq / num_features + eps);

    // Normalize across the feature dimension
    for (int feat = 0; feat < num_features; feat++) {
        int idx = batch_offset + feat * numel_per_batch + offset;
        output[idx] = input[idx] / rms;
    }
}

// CUDA forward function using 2D grid to ensure coalesced memory accesses

torch::Tensor rms_norm_cuda_forward_2d_coalesced(torch::Tensor input, float eps) {
    // Allocate output tensor of the same shape as input
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    // Compute numel per batch from dimensions beyond the first two
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Define block size along the spatial (numel_per_batch) dimension
    const int threads_per_block = 256;
    int blocks_x = (numel_per_batch + threads_per_block - 1) / threads_per_block;
    // Grid dimensions: blocks_x covers spatial dimension; batch_size covers batch dimension
    dim3 blocks(blocks_x, batch_size);
    dim3 threads(threads_per_block);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_2d_coalesced", ([&] {
        rms_norm_2d_coalesced_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_features,
            numel_per_batch,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward_2d_coalesced, "RMS normalization forward with 2D coalesced memory accesses (CUDA)");
}
