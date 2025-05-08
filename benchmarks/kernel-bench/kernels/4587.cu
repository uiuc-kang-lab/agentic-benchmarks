#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use shared memory to store intermediate sums that threads within a block can collaborate on

// Define CUDA kernel using shared memory
template <typename scalar_t>
__global__ void rms_norm_kernel_shared_memory(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    extern __shared__ float s_sumsq[];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = tid / numel_per_batch;
    
    if (batch_id >= batch_size) return;
    
    const int offset_in_batch = tid % numel_per_batch;
    const int batch_offset = batch_id * num_features * numel_per_batch;

    // Initialize shared memory
    float local_sumsq = 0.0;

    // Compute sum of squares using shared memory for the entire warp
    for (int feat = 0; feat < num_features; feat++) {
        const float val = input[batch_offset + feat * numel_per_batch + offset_in_batch];
        local_sumsq += val * val;
    }

    // Thread 0 of each block adds local sums to shared memory and computes the final sum
    atomicAdd(&s_sumsq[threadIdx.x], local_sumsq);

    // Ensure all updates are done before computing RMS
    __syncthreads();
    
    // Compute RMS from shared memory
    const float rms = sqrt(s_sumsq[threadIdx.x] / num_features + eps);

    // Normalize using computed RMS value
    for (int feat = 0; feat < num_features; feat++) {
        const int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
        output[idx] = input[idx] / rms;
    }
}

torch::Tensor rms_norm_cuda_forward_shared_memory(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    // Calculate elements per batch for dimensions beyond first two
    int numel_per_batch = 1;
    for(int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    const int total_threads = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_shared_memory", ([&] {
        rms_norm_kernel_shared_memory<scalar_t><<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(
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
    m.def("forward", &rms_norm_cuda_forward_shared_memory, "RMS normalization forward with shared memory (CUDA)");
}
