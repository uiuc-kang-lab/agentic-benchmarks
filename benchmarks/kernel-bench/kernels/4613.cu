/*
 * Combined CUDA kernel for RMS normalization using constant memory for precomputed feature offsets
 * and loop unrolling for uniform control flow.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <stdexcept>

#define MAX_FEATURES_CONST 1024
__constant__ int d_offsets[MAX_FEATURES_CONST];

// Kernel that leverages constant memory for feature offsets and loop unrolling in inner loops

template <typename scalar_t>
__global__ void rms_norm_kernel_const_unroll(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    for (; idx < batch_size * numel_per_batch; idx += total_threads) {
        const int batch_id = idx / numel_per_batch;
        const int offset_in_batch = idx % numel_per_batch;
        const int batch_offset = batch_id * num_features * numel_per_batch;

        scalar_t sumsq = 0;
        // Loop unrolling for sum of squares computation
        #pragma unroll
        for (int feat = 0; feat < num_features; feat++) {
            int pos = batch_offset + d_offsets[feat] + offset_in_batch;
            scalar_t val = input[pos];
            sumsq += val * val;
        }

        scalar_t rms = sqrt(sumsq / num_features + eps);

        // Loop unrolling for normalization
        #pragma unroll
        for (int feat = 0; feat < num_features; feat++) {
            int pos = batch_offset + d_offsets[feat] + offset_in_batch;
            output[pos] = input[pos] / rms;
        }
    }
}

// CUDA forward function that prepares constant memory and launches the optimized kernel

torch::Tensor rms_norm_cuda_forward_const_unroll(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Ensure num_features does not exceed constant memory limit
    if (num_features > MAX_FEATURES_CONST) {
        throw std::runtime_error("num_features exceeds the constant memory limit.");
    }

    // Precompute feature offsets and copy them to constant memory
    std::vector<int> offsets(num_features);
    for (int i = 0; i < num_features; i++) {
        offsets[i] = i * numel_per_batch;
    }
    cudaMemcpyToSymbol(d_offsets, offsets.data(), num_features * sizeof(int));

    const int total_threads = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_const_unroll", ([&] {
        rms_norm_kernel_const_unroll<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward_const_unroll, "RMS normalization forward with constant memory and loop unrolling optimization (CUDA)");
}
