#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function to compute sum of squares for a given batch element and offset
template <typename scalar_t>
__device__ inline scalar_t compute_sum_squares(const scalar_t* __restrict__ input,
                                                int base,
                                                int num_features,
                                                int offset,
                                                int numel_per_batch) {
    scalar_t sumsq = static_cast<scalar_t>(0);
    #pragma unroll
    for (int feat = 0; feat < num_features; feat++) {
        int idx = base + feat * numel_per_batch + offset;
        scalar_t val = input[idx];
        sumsq += val * val;
    }
    return sumsq;
}

// Device function to normalize input values and store the result in output
template <typename scalar_t>
__device__ inline void normalize_output(const scalar_t* __restrict__ input,
                                          scalar_t* __restrict__ output,
                                          int base,
                                          int num_features,
                                          int offset,
                                          int numel_per_batch,
                                          scalar_t rms) {
    #pragma unroll
    for (int feat = 0; feat < num_features; feat++) {
        int idx = base + feat * numel_per_batch + offset;
        output[idx] = input[idx] / rms;
    }
}

// Modularized RMSNorm CUDA kernel using helper device functions
template <typename scalar_t>
__global__ void modular_rms_norm_kernel(const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ output,
                                         const int batch_size,
                                         const int num_features,
                                         const int numel_per_batch,
                                         const float eps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int index = tid; index < batch_size * numel_per_batch; index += total_threads) {
        int batch_id = index / numel_per_batch;
        int offset = index % numel_per_batch;
        int base = batch_id * num_features * numel_per_batch;

        scalar_t sumsq = compute_sum_squares<scalar_t>(input, base, num_features, offset, numel_per_batch);
        scalar_t rms = sqrt(sumsq / num_features + eps);
        normalize_output<scalar_t>(input, output, base, num_features, offset, numel_per_batch, rms);
    }
}

// CUDA forward function
torch::Tensor rms_norm_cuda_forward_modular(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    // Compute number of elements per batch for dimensions beyond the first two
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    const int total_work = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    const int blocks = (total_work + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_modular", ([&] {
        modular_rms_norm_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward_modular, "Modularized RMS normalization forward (CUDA)");
}
