#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Optimized CUDA kernel using stride loops for even workload distribution.
// Each thread handles multiple data elements spaced by the total number of threads.
// Boundary conditions are correctly handled through the stride loops.

template <typename scalar_t>
__global__ void rms_norm_kernel_stride_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int total_work,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple elements by incrementing its index by the stride
    for (int i = tid; i < total_work; i += stride) {
        // Compute batch index and offset within the batch
        int batch_id = i / numel_per_batch;
        int offset = i % numel_per_batch;
        int base = batch_id * num_features * numel_per_batch;

        // Compute the sum of squares over the feature dimension
        scalar_t sumsq = static_cast<scalar_t>(0);
        for (int feat = 0; feat < num_features; feat++) {
            int idx = base + feat * numel_per_batch + offset;
            scalar_t val = input[idx];
            sumsq += val * val;
        }

        // Compute RMS with epsilon for numerical stability
        scalar_t rms = sqrt(sumsq / num_features + eps);

        // Normalize the input values
        for (int feat = 0; feat < num_features; feat++) {
            int idx = base + feat * numel_per_batch + offset;
            output[idx] = input[idx] / rms;
        }
    }
}

// CUDA forward function

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    // Compute number of elements per batch for dimensions beyond the first two
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Total work items correspond to each element offset in a batch
    int total_work = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    int blocks = (total_work + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_stride_optimized", ([&] {
        rms_norm_kernel_stride_optimized<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward with optimized stride loop (CUDA)");
}
