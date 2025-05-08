#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rms_norm_kernel_stride(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // Process multiple work units in strides
    for (int linear_id = tid; linear_id < batch_size * numel_per_batch; linear_id += total_threads) {
        const int batch_id = linear_id / numel_per_batch;
        const int offset_in_batch = linear_id % numel_per_batch;
        const int batch_offset = batch_id * num_features * numel_per_batch;
        
        if (batch_id >= batch_size) return;

        // Calculate sum of squares
        scalar_t sumsq = 0.0f;
        const scalar_t inv_num_features = 1.0f / num_features;  // Precompute division
        
        // Cache the input values and compute sum of squares in one pass
        scalar_t* cached_vals = new scalar_t[num_features];
        for (int feat = 0; feat < num_features; feat++) {
            const int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
            cached_vals[feat] = input[idx];
            sumsq += cached_vals[feat] * cached_vals[feat];
        }
        
        // Calculate RMS (using precomputed inverse)
        const scalar_t rms = sqrt(sumsq * inv_num_features + eps);
        const scalar_t inv_rms = 1.0f / rms;  // Precompute division
        
        // Normalize using cached values and precomputed inverse
        for (int feat = 0; feat < num_features; feat++) {
            const int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
            output[idx] = cached_vals[feat] * inv_rms;
        }
        
        delete[] cached_vals;
    }
}

torch::Tensor rms_norm_cuda_forward_stride(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    // Calculate elements per batch for dimensions beyond first two
    int numel_per_batch = 1;
    for(int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    const int total_threads = 1024; // Using more threads
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_stride", ([&] {
        rms_norm_kernel_stride<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward_stride, "RMS normalization forward (CUDA) with stride optimization");
}