#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel utilizing shared memory to store intermediate sums reducing global memory latency.

// Each block processes one batch entirely to utilize shared memory efficiently
// Shared memory is used to store sum of squares across features within a block

template <typename scalar_t>
__global__ void rms_norm_kernel_shared_memory(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    extern __shared__ scalar_t shared_sumsq[];
    int tid = threadIdx.x;

    int batch_id = blockIdx.x;

    // Compute offset within the batch
    int offset_in_batch = tid;
    int batch_offset = batch_id * num_features * numel_per_batch;

    // Initialize shared memory to zero
    shared_sumsq[tid] = 0.0f;
    __syncthreads();

    // Each thread calculates the sum of squares for its portion
    for (int feat = 0; feat < num_features; feat++) {
        int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
        scalar_t val = input[idx];
        atomicAdd(&shared_sumsq[tid], val * val);
    }
    __syncthreads();

    // Reduce sum within block using the first thread
    scalar_t sumsq = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            sumsq += shared_sumsq[i];
        }
    }
    __syncthreads();

    // Broadcast reduced sum to all threads in the block
    sumsq = __shfl_sync(0xFFFFFFFF, sumsq, 0);

    // Calculate RMS
    scalar_t rms = sqrt(sumsq / num_features + eps);

    // Normalize input data
    for (int feat = 0; feat < num_features; feat++) {
        int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
        output[idx] = input[idx] / rms;
    }
}

torch::Tensor rms_norm_cuda_forward_shared_memory(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    // Calculate elements per batch
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Launch one block per batch with shared memory
    const int threads_per_block = numel_per_batch;
    const int blocks = batch_size;
    const int shared_memory_size = threads_per_block * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_shared_memory", ([&] {
        rms_norm_kernel_shared_memory<scalar_t><<<blocks, threads_per_block, shared_memory_size>>>(
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
    m.def("forward", &rms_norm_cuda_forward_shared_memory, "RMS normalization forward with shared memory (CUDA)");
}
