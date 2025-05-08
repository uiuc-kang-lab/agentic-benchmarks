#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// This kernel uses __ldg() for optimized global memory read accesses.
// It assumes that both input and output pointers are 128-bit aligned to leverage coalesced access
// and the GPU's read-only data cache. This can reduce latency on the NVIDIA H100 GPU.

template <typename scalar_t>
__global__ void rms_norm_kernel_ldg_aligned(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Each thread processes multiple work-items via striding
    for (int index = tid; index < batch_size * numel_per_batch; index += total_threads) {
        int batch_id = index / numel_per_batch;
        int offset = index % numel_per_batch;
        int base = batch_id * num_features * numel_per_batch;

        // Compute the sum of squares using __ldg() for read-only, aligned accesses
        scalar_t sumsq = static_cast<scalar_t>(0);
        for (int feat = 0; feat < num_features; feat++) {
            int pos = base + feat * numel_per_batch + offset;
            scalar_t val = __ldg(&input[pos]);
            sumsq += val * val;
        }
        
        // Compute RMS with epsilon for numerical stability
        scalar_t rms = sqrt(sumsq / num_features + eps);

        // Normalize the input using __ldg() to fetch the original values
        for (int feat = 0; feat < num_features; feat++) {
            int pos = base + feat * numel_per_batch + offset;
            scalar_t val = __ldg(&input[pos]);
            output[pos] = val / rms;
        }
    }
}

// CUDA forward function that sets up the kernel launch parameters

torch::Tensor rms_norm_cuda_forward_ldg_aligned(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    int total_work = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    const int blocks = (total_work + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_ldg_aligned", ([&] {
        rms_norm_kernel_ldg_aligned<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward_ldg_aligned, "RMS normalization forward using __ldg() for aligned memory accesses (CUDA)");
}
