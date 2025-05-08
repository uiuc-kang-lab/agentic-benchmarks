#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel: each (batch, offset) pair is processed by one warp using warp-level reduction

template <typename scalar_t>
__global__ void rms_norm_warp_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    // Calculate global warp ID and lane within the warp
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize;  // each warp handles one work item
    int lane = global_thread_id % warpSize;

    // Total number of (batch, offset) pairs to process
    int total_work = batch_size * numel_per_batch;
    if (warp_id >= total_work) return;

    // Determine the batch and offset indices
    int batch_id = warp_id / numel_per_batch;
    int offset = warp_id % numel_per_batch;
    int base = batch_id * num_features * numel_per_batch;

    // Each thread in the warp computes partial sum over features assigned in a strided loop
    scalar_t sum = 0;
    for (int feat = lane; feat < num_features; feat += warpSize) {
        int idx = base + feat * numel_per_batch + offset;
        scalar_t val = input[idx];
        sum += val * val;
    }

    // Warp-level reduction using shuffle intrinsics
    for (int offset_shuffle = warpSize / 2; offset_shuffle > 0; offset_shuffle /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset_shuffle);
    }

    // Lane 0 computes the RMS value
    scalar_t rms_val;
    if (lane == 0) {
        rms_val = sqrt(sum / num_features + eps);
    }
    // Broadcast the RMS value to all lanes in the warp
    rms_val = __shfl_sync(0xffffffff, rms_val, 0);

    // Normalize: each thread processes a subset of features
    for (int feat = lane; feat < num_features; feat += warpSize) {
        int idx = base + feat * numel_per_batch + offset;
        scalar_t val = input[idx];
        output[idx] = val / rms_val;
    }
}

// CUDA forward function using warp-level reduction

torch::Tensor rms_norm_cuda_forward_warp_reduce(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    int batch_size = input.size(0);
    int num_features = input.size(1);

    // Calculate number of elements per batch for dimensions beyond the first two
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Total work items = batch_size * numel_per_batch, each work item is processed by one warp
    int total_work = batch_size * numel_per_batch;
    int total_threads = total_work * warpSize; // warpSize is 32

    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_warp_reduce", ([&] {
        rms_norm_warp_reduce_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward_warp_reduce, "RMS normalization forward with warp-level reduction (CUDA)");
}
