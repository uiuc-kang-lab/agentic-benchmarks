#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel using a tuned block size configuration (512 threads per block) based on experimental tuning.
// The grid is organized in 2D: x-dimension for contiguous elements within a batch and y-dimension for batch index.

template <typename scalar_t>
__global__ void rms_norm_kernel_tuned(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    // Each block row corresponds to a batch
    int batch_id = blockIdx.y;
    
    // x-dimension for contiguous offset within a batch
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= numel_per_batch) return;

    int batch_offset = batch_id * num_features * numel_per_batch;

    // Compute the sum of squares for the current element offset
    scalar_t sumsq = static_cast<scalar_t>(0);
    for (int feat = 0; feat < num_features; feat++) {
        int idx = batch_offset + feat * numel_per_batch + offset;
        scalar_t val = input[idx];
        sumsq += val * val;
    }

    // Calculate RMS value ensuring numerical stability with eps
    scalar_t rms = sqrt(sumsq / num_features + eps);

    // Normalize the values by dividing by the RMS
    for (int feat = 0; feat < num_features; feat++) {
        int idx = batch_offset + feat * numel_per_batch + offset;
        output[idx] = input[idx] / rms;
    }
}

// CUDA forward function with tuned block size

torch::Tensor rms_norm_cuda_forward_tuned(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    // Compute the number of elements per batch (for dims beyond the first two)
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Experimentally tuned block size (try with 32, 64, 128, 256, 512) -- here we use 512 based on tuning for H100
    constexpr int threads_per_block = 512;
    int grid_x = (numel_per_batch + threads_per_block - 1) / threads_per_block;
    dim3 blocks(grid_x, batch_size);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_tuned", ([&] {
        rms_norm_kernel_tuned<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward_tuned, "RMS normalization forward (CUDA) with tuned block size");
}
