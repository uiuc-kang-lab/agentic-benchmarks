#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function to compute the sum of squares across features
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_sum_squares(const scalar_t* __restrict__ input,
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

// Device function to normalize input elements using the computed RMS value
template <typename scalar_t>
__device__ __forceinline__ void normalize_element(const scalar_t* __restrict__ input,
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

// Modularized RMSNorm CUDA kernel using inline device functions
template <typename scalar_t>
__global__ void rms_norm_kernel_modular(const scalar_t* __restrict__ input,
                                          scalar_t* __restrict__ output,
                                          int batch_size,
                                          int num_features,
                                          int numel_per_batch,
                                          float eps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * numel_per_batch;
    int stride = blockDim.x * gridDim.x;

    for (int index = tid; index < total_elements; index += stride) {
        int batch_id = index / numel_per_batch;
        int offset = index % numel_per_batch;
        int base = batch_id * num_features * numel_per_batch;

        scalar_t sumsq = compute_sum_squares<scalar_t>(input, base, num_features, offset, numel_per_batch);
        scalar_t rms = sqrt(sumsq / num_features + eps);
        normalize_element<scalar_t>(input, output, base, num_features, offset, numel_per_batch, rms);
    }
}

// CUDA forward function that launches the modular kernel
torch::Tensor rms_norm_cuda_forward_modular_inlined(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    int batch_size = input.size(0);
    int num_features = input.size(1);
    
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    int total_elements = batch_size * numel_per_batch;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_modular_inlined", ([&] {
        rms_norm_kernel_modular<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward_modular_inlined, "Modular and inlined RMS normalization forward (CUDA)");
}
