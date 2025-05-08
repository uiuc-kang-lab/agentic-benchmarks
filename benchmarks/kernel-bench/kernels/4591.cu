#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel using a stride loop and a tuned block size (128 threads per block) based on experimental evaluation
// for improved occupancy and reduced runtime on NVIDIA H100 GPU with CUDA 12.2.

template <typename scalar_t>
__global__ void rms_norm_kernel_block_tune(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    const int total_work = batch_size * numel_per_batch;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int index = tid; index < total_work; index += stride) {
        int batch_id = index / numel_per_batch;
        int offset_in_batch = index % numel_per_batch;
        int batch_offset = batch_id * num_features * numel_per_batch;

        // Calculate sum of squares across features
        scalar_t sumsq = 0.0;
        for (int feat = 0; feat < num_features; feat++) {
            int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
            scalar_t val = input[idx];
            sumsq += val * val;
        }

        // Compute RMS with epsilon for numerical stability
        scalar_t rms = sqrt(sumsq / num_features + eps);

        // Normalize each feature value by dividing by the RMS
        for (int feat = 0; feat < num_features; feat++) {
            int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
            output[idx] = input[idx] / rms;
        }
    }
}

// CUDA forward function with block size tuned to 128 threads per block

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    // Calculate number of elements per batch for dimensions beyond the first two
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    int total_work = batch_size * numel_per_batch;
    constexpr int threads_per_block = 128;  // Tuned block size based on experimental evaluation
    int blocks = (total_work + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_block_tune", ([&] {
        rms_norm_kernel_block_tune<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward with tuned block size (CUDA)");
}
