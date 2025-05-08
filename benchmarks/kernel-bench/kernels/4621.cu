#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHUNK_SIZE 128

// Kernel 1: Compute partial sum of squares using atomicAdd in global memory
// Each work item processes a chunk of features for a specific (batch, offset) pair

template <typename scalar_t>
__global__ void rms_norm_atomic_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ sumsqs,
    const int batch_size,
    const int num_features,
    const int numel_per_batch
) {
    int total_offsets = batch_size * numel_per_batch;
    int num_chunks = (num_features + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work = total_offsets * num_chunks;
    if (global_idx >= total_work) return;

    // Map global_idx to a unique (batch, offset) pair and a feature chunk index
    int off_idx = global_idx / num_chunks; // index for (batch, offset) pair
    int chunk   = global_idx % num_chunks;

    int batch_id = off_idx / numel_per_batch;
    int offset   = off_idx % numel_per_batch;

    int feat_start = chunk * CHUNK_SIZE;
    int feat_end = (feat_start + CHUNK_SIZE < num_features) ? (feat_start + CHUNK_SIZE) : num_features;

    scalar_t local_sum = 0;
    int base = batch_id * num_features * numel_per_batch;
    for (int feat = feat_start; feat < feat_end; feat++) {
         int pos = base + feat * numel_per_batch + offset;
         scalar_t val = input[pos];
         local_sum += val * val;
    }
    // Use atomicAdd to accumulate partial sum into the global temporary array
    atomicAdd(&sumsqs[off_idx], local_sum);
}

// Kernel 2: Normalize the input tensor using the computed sums of squares

template <typename scalar_t>
__global__ void rms_norm_atomic_normalize_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ sumsqs,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    int total_elements = batch_size * num_features * numel_per_batch;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (; idx < total_elements; idx += stride) {
         int batch_id = idx / (num_features * numel_per_batch);
         int rem = idx % (num_features * numel_per_batch);
         int feat = rem / numel_per_batch;
         int offset = rem % numel_per_batch;
         int off_idx = batch_id * numel_per_batch + offset;
         scalar_t sumsq = sumsqs[off_idx];
         scalar_t rms = sqrt(sumsq / static_cast<scalar_t>(num_features) + static_cast<scalar_t>(eps));
         output[idx] = input[idx] / rms;
    }
}

// Host function that launches the two-phase kernel

torch::Tensor rms_norm_cuda_forward_atomic(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    // Compute elements per batch for dimensions beyond the first two
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    int total_offsets = batch_size * numel_per_batch;
    
    // Allocate temporary tensor to store sums of squares for each (batch, offset) pair
    auto sumsqs = torch::zeros({total_offsets}, input.options());

    // Launch reduction kernel
    int num_chunks = (num_features + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int total_work = total_offsets * num_chunks;
    const int threads_reduce = 256;
    int blocks_reduce = (total_work + threads_reduce - 1) / threads_reduce;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_atomic_reduce", ([&] {
        rms_norm_atomic_reduce_kernel<scalar_t><<<blocks_reduce, threads_reduce>>>(
            input.data_ptr<scalar_t>(),
            sumsqs.data_ptr<scalar_t>(),
            batch_size,
            num_features,
            numel_per_batch
        );
    }));

    // Launch normalization kernel
    int total_elements = batch_size * num_features * numel_per_batch;
    const int threads_normalize = 256;
    int blocks_normalize = (total_elements + threads_normalize - 1) / threads_normalize;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_atomic_normalize", ([&] {
        rms_norm_atomic_normalize_kernel<scalar_t><<<blocks_normalize, threads_normalize>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            sumsqs.data_ptr<scalar_t>(),
            batch_size,
            num_features,
            numel_per_batch,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward_atomic, "RMS normalization forward with atomic two-phase reduction (CUDA)");
}
