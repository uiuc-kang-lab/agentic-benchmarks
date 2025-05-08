#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel with optimized block size for improved performance

template <typename scalar_t>
__global__ void optimized_rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    for (int index = tid; index < batch_size * numel_per_batch; index += total_threads) {
        const int batch_id = index / numel_per_batch;
        const int offset_in_batch = index % numel_per_batch;
        const int batch_offset = batch_id * num_features * numel_per_batch;

        // Calculate sum of squares
        scalar_t sumsq = 0.0f;
        #pragma unroll
        for (int feat = 0; feat < num_features; feat++) {
            const scalar_t val = input[batch_offset + feat * numel_per_batch + offset_in_batch];
            sumsq += val * val;
        }

        // Calculate RMS
        const scalar_t rms = sqrt(sumsq / num_features + eps);

        // Normalize
        #pragma unroll
        for (int feat = 0; feat < num_features; feat++) {
            const int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
            output[idx] = input[idx] / rms;
        }
    }
}

// CUDA forward function with optimized block size

torch::Tensor rms_norm_cuda_forward_optimized(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    // Calculate elements per batch for dimensions beyond first two
    int numel_per_batch = 1;
    for(int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    const int total_threads = batch_size * numel_per_batch;
    const int threads_per_block = 512; // Optimized block size
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_optimized", ([&] {
        optimized_rms_norm_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward_optimized, "RMS normalization forward with optimized block size (CUDA)");
}
