#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// New kernel launching with a 2D grid to ensure threads in a warp belong to the same batch
// and access consecutive memory locations along the innermost dimension.

template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    // Each block row processes one batch
    int batch_id = blockIdx.y;
    // x-dimension indexes the contiguous offset inside each batch
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= numel_per_batch) return;

    const int batch_offset = batch_id * num_features * numel_per_batch;
    scalar_t sumsq = 0.0f;

    // Loop over features; note that for each feature the memory address is:
    // batch_offset + feat * numel_per_batch + offset, which means that threads
    // in the same warp (with consecutive offset values) access consecutive memory locations
    for (int feat = 0; feat < num_features; feat++) {
        int idx = batch_offset + feat * numel_per_batch + offset;
        scalar_t val = input[idx];
        sumsq += val * val;
    }

    // Compute RMS with epsilon for numerical stability
    scalar_t rms = sqrt(sumsq / num_features + eps);

    // Normalize the input and store to output using coalesced accesses
    for (int feat = 0; feat < num_features; feat++) {
        int idx = batch_offset + feat * numel_per_batch + offset;
        output[idx] = input[idx] / rms;
    }
}

// CUDA forward function

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    // Compute the number of elements per batch beyond the first two dimensions
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Launch a 2D kernel:
    //   - x-dimension covers the numel_per_batch (ensuring coalesced accesses within a batch)
    //   - y-dimension indexes over batches
    int threads_per_block = 256;
    dim3 blocks((numel_per_batch + threads_per_block - 1) / threads_per_block, batch_size);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward (CUDA)");
}
