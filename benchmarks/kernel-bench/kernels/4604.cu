#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel with manual loop unrolling using #pragma unroll

template <typename scalar_t>
__global__ void rms_norm_kernel_unroll(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int total_work,    // batch_size * numel_per_batch
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple work-items in strides
    for (int index = tid; index < total_work; index += stride) {
        const int batch_id = index / numel_per_batch;
        const int offset = index % numel_per_batch;
        const int batch_offset = batch_id * num_features * numel_per_batch;

        scalar_t sumsq = static_cast<scalar_t>(0);

        // Unroll the loop for summing the squares
        #pragma unroll
        for (int feat = 0; feat < num_features; feat++) {
            int idx = batch_offset + feat * numel_per_batch + offset;
            scalar_t val = input[idx];
            sumsq += val * val;
        }

        // Compute the RMS value with epsilon for stability
        scalar_t rms = sqrt(sumsq / num_features + eps);

        // Unroll the loop for normalizing the values
        #pragma unroll
        for (int feat = 0; feat < num_features; feat++) {
            int idx = batch_offset + feat * numel_per_batch + offset;
            output[idx] = input[idx] / rms;
        }
    }
}

// CUDA forward function

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
    const int threads_per_block = 256;
    const int blocks = (total_work + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_unroll", ([&] {
        rms_norm_kernel_unroll<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_work,
            num_features,
            numel_per_batch,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward with loop unrolling (CUDA)");
}
