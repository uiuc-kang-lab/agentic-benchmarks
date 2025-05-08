#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel uses a 2D grid: one dimension for batches and one for offset groups within each batch.
// Each thread processes one or more offsets (from the contiguous dimensions) for a given batch.

template <typename scalar_t>
__global__ void rms_norm_kernel_2d(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    // Each block in the x-dimension handles one batch
    int batch_id = blockIdx.x;
    if (batch_id >= batch_size) return;
    
    // Each block in the y-dimension covers a subset of offsets (elements in the trailing dimensions)
    int offset_start = blockIdx.y * blockDim.x;
    
    // Process offsets in a grid-stride loop along the y-dimension
    for (int offset = offset_start + threadIdx.x; offset < numel_per_batch; offset += gridDim.y * blockDim.x) {
        // Compute the starting index for the current batch
        int batch_offset = batch_id * num_features * numel_per_batch;

        // Calculate sum of squares over features for this offset
        scalar_t sumsq = 0;
        for (int feat = 0; feat < num_features; feat++) {
            int idx = batch_offset + feat * numel_per_batch + offset;
            scalar_t val = input[idx];
            sumsq += val * val;
        }

        // Compute RMS with epsilon for numerical stability
        scalar_t rms = sqrt(sumsq / num_features + eps);

        // Normalize each feature's value
        for (int feat = 0; feat < num_features; feat++) {
            int idx = batch_offset + feat * numel_per_batch + offset;
            output[idx] = input[idx] / rms;
        }
    }
}


// The forward function sets up a 2D grid where the x-dimension maps to batches,
// and the y-dimension splits the numel_per_batch domain into chunks of size blockDim.x

torch::Tensor rms_norm_cuda_forward_2d(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    // Determine number of elements per batch from dimensions beyond the first two
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Choose 256 threads per block for the offset dimension
    const int threads_per_block = 256;
    // Determine how many blocks are needed to cover all offsets
    int grid_y = (numel_per_batch + threads_per_block - 1) / threads_per_block;
    
    // Set up a 2D grid: x-dimension for batch, y-dimension for offset groups
    dim3 blocks(batch_size, grid_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_forward_2d", ([&] {
        rms_norm_kernel_2d<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_features,
            numel_per_batch,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward_2d, "RMS normalization forward with 2D grid (CUDA)");
}
