#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Declare constant memory for frequently accessed read-only parameters
__constant__ int CONST_NUM_FEATURES;
__constant__ int CONST_NUMEL_PER_BATCH;
__constant__ float CONST_EPS;

// CUDA kernel that uses constant memory to hold parameters
template <typename scalar_t>
__global__ void rms_norm_kernel_constant(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size  // batch_size remains a kernel parameter
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    // Total work: each work-item corresponds to one offset within a batch
    const int total_work = batch_size * CONST_NUMEL_PER_BATCH;

    for (int index = tid; index < total_work; index += total_threads) {
        int batch_id = index / CONST_NUMEL_PER_BATCH;
        int offset_in_batch = index % CONST_NUMEL_PER_BATCH;
        int batch_offset = batch_id * CONST_NUM_FEATURES * CONST_NUMEL_PER_BATCH;

        // Compute the sum of squares across the feature dimension
        scalar_t sumsq = 0;
        for (int feat = 0; feat < CONST_NUM_FEATURES; feat++) {
            int idx = batch_offset + feat * CONST_NUMEL_PER_BATCH + offset_in_batch;
            scalar_t val = input[idx];
            sumsq += val * val;
        }

        // Calculate RMS normalization factor
        scalar_t rms = sqrt(sumsq / CONST_NUM_FEATURES + CONST_EPS);

        // Normalize input elements
        for (int feat = 0; feat < CONST_NUM_FEATURES; feat++) {
            int idx = batch_offset + feat * CONST_NUMEL_PER_BATCH + offset_in_batch;
            output[idx] = input[idx] / rms;
        }
    }
}

// CUDA forward function

torch::Tensor rms_norm_cuda_forward_constant(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    // Get dimensions
    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    // Compute number of elements per batch for dimensions beyond the first two
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Copy frequently accessed parameters to constant memory
    cudaMemcpyToSymbol(CONST_NUM_FEATURES, &num_features, sizeof(int));
    cudaMemcpyToSymbol(CONST_NUMEL_PER_BATCH, &numel_per_batch, sizeof(int));
    cudaMemcpyToSymbol(CONST_EPS, &eps, sizeof(float));

    // Launch settings
    const int total_work = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    int blocks = (total_work + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_constant", ([&] {
        rms_norm_kernel_constant<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward_constant, "RMS normalization forward with constant memory (CUDA)");
}
