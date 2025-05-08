#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses block-level parallel reduction with shared memory to compute the sum of squares
// across the features dimension for each (batch, offset) group. Each block processes one group
// and writes the normalized output. No global atomic operations are used, minimizing contention.

template <typename scalar_t>
__global__ void rms_norm_kernel_parallel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int num_groups,         // num_groups = batch_size * numel_per_batch
    const int num_features,
    const int numel_per_batch,    // used to compute correct indexing
    const float eps
) {
    // Each block processes one group corresponding to a (batch, offset) pair
    int group = blockIdx.x;
    if (group >= num_groups) return;
    int batch_id = group / numel_per_batch;
    int offset = group % numel_per_batch;
    int batch_offset = batch_id * num_features * numel_per_batch;

    // Use shared memory for reduction; size is blockDim.x
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Each thread computes a partial sum over the features assigned to it
    scalar_t partial_sum = 0;
    for (int feat = threadIdx.x; feat < num_features; feat += blockDim.x) {
        int idx = batch_offset + feat * numel_per_batch + offset;
        scalar_t val = input[idx];
        partial_sum += val * val;
    }
    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 computes the RMS
    scalar_t rms = 0;
    if (threadIdx.x == 0) {
        rms = sqrt(sdata[0] / num_features + eps);
        sdata[0] = rms;  // store in shared memory for broadcast
    }
    __syncthreads();
    rms = sdata[0];

    // All threads use the computed rms to normalize a subset of features
    for (int feat = threadIdx.x; feat < num_features; feat += blockDim.x) {
        int idx = batch_offset + feat * numel_per_batch + offset;
        output[idx] = input[idx] / rms;
    }
}


torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Each group corresponds to one offset within a batch
    const int num_groups = batch_size * numel_per_batch;

    // Choose a block size that is appropriate for reduction (e.g., 256 threads)
    const int threads_per_block = 256;
    // Launch one block per group. Shared memory size = threads_per_block * sizeof(scalar_t).
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel_parallel<scalar_t><<<
            num_groups,
            threads_per_block,
            threads_per_block * sizeof(scalar_t)
        >>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_groups,
            num_features,
            numel_per_batch,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward (CUDA) using shared memory reduction");
}
