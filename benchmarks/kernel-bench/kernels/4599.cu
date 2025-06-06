#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <stdexcept>

#define MAX_FEATURES_CONST 1024
__constant__ int d_offsets[MAX_FEATURES_CONST];

// Kernel that uses constant memory to store precomputed feature offsets
template <typename scalar_t>
__global__ void rms_norm_kernel_const(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Each thread processes multiple elements
    for (; idx < batch_size * numel_per_batch; idx += total_threads) {
        int batch_id = idx / numel_per_batch;
        int offset_in_batch = idx % numel_per_batch;
        int batch_offset = batch_id * num_features * numel_per_batch;
        
        // Compute sum of squares using precomputed offsets from constant memory
        scalar_t sumsq = 0;
        for (int feat = 0; feat < num_features; feat++) {
            int pos = batch_offset + d_offsets[feat] + offset_in_batch;
            scalar_t val = input[pos];
            sumsq += val * val;
        }
        
        scalar_t rms = sqrt(sumsq / num_features + eps);
        
        // Normalize the input values
        for (int feat = 0; feat < num_features; feat++) {
            int pos = batch_offset + d_offsets[feat] + offset_in_batch;
            output[pos] = input[pos] / rms;
        }
    }
}

// CUDA forward function that copies precomputed offsets to constant memory
torch::Tensor rms_norm_cuda_forward_const(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    // Compute elements per batch for dimensions beyond the first two
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }
    
    // Ensure num_features fits in our constant memory allocation
    if (num_features > MAX_FEATURES_CONST) {
        throw std::runtime_error("num_features exceeds the constant memory limit.");
    }
    
    // Prepare host-side offsets: offset for each feature is feat * numel_per_batch
    std::vector<int> offsets(num_features);
    for (int i = 0; i < num_features; i++) {
        offsets[i] = i * numel_per_batch;
    }
    
    // Copy the offsets to constant memory
    cudaMemcpyToSymbol(d_offsets, offsets.data(), num_features * sizeof(int));
    
    const int total_threads = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_const", ([&] {
        rms_norm_kernel_const<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward_const, "RMS normalization forward with constant memory optimization (CUDA)");
}
