#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define UNROLL 4 // Unroll factor

// Kernel optimized for minimal warp divergence

template <typename scalar_t>
__global__ void rms_norm_kernel_warp_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    extern __shared__ float sdata[];  // Shared memory
    scalar_t* shared_sum = sdata;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int index = tid; index < batch_size * numel_per_batch; index += stride) {
        const int batch_id = index / numel_per_batch;
        const int offset_in_batch = index % numel_per_batch;
        const int batch_offset = batch_id * num_features * numel_per_batch;

        // Calculate sum of squares using loop unrolling
        scalar_t sumsq = 0.0f;
        for (int feat = 0; feat < num_features; feat += UNROLL) {
            #pragma unroll
            for (int i = 0; i < UNROLL && (feat + i) < num_features; ++i) {
                const scalar_t val = input[batch_offset + (feat + i) * numel_per_batch + offset_in_batch];
                sumsq += val * val;
            }
        }

        if (threadIdx.x == 0) {
            shared_sum[blockIdx.x] = sumsq;
        }
        __syncthreads();

        if (tid < gridDim.x) {
            sumsq = shared_sum[blockIdx.x];
            const scalar_t rms = sqrt(sumsq / num_features + eps);

            // Normalize and store results
            for (int feat = 0; feat < num_features; feat += UNROLL) {
                #pragma unroll
                for (int i = 0; i < UNROLL && (feat + i) < num_features; ++i) {
                    const int idx = batch_offset + (feat + i) * numel_per_batch + offset_in_batch;
                    output[idx] = input[idx] / rms;
                }
            }
        }
        __syncthreads();
    }
}

// CUDA forward function

torch::Tensor rms_norm_cuda_forward_warp_optimized(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    const int total_threads = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_warp_optimized", ([&] {
        rms_norm_kernel_warp_optimized<scalar_t><<<blocks, threads_per_block, blocks * sizeof(float)>>>(
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
    m.def("forward", &rms_norm_cuda_forward_warp_optimized, "RMS normalization forward optimized for warp (CUDA)");
}