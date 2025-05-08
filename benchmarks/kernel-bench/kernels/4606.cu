#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel using shared memory for efficiency and minimal synchronization

template <typename scalar_t>
__global__ void rms_norm_kernel_shared(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    extern __shared__ scalar_t sdata[];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    for (int index = tid; index < batch_size * numel_per_batch; index += total_threads) {
        const int batch_id = index / numel_per_batch;
        const int offset_in_batch = index % numel_per_batch;
        const int batch_offset = batch_id * num_features * numel_per_batch;

        // Load data into shared memory
        scalar_t sumsq = 0.0f;
        #pragma unroll
        for (int feat = 0; feat < num_features; feat++) {
            const scalar_t val = input[batch_offset + feat * numel_per_batch + offset_in_batch];
            sumsq += val * val;
        }

        // Store partial sum in shared memory
        sdata[threadIdx.x] = sumsq;
        __syncthreads();

        // Perform reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }

        // Calculate RMS and normalize
        if (threadIdx.x == 0) {
            scalar_t rms = sqrt(sdata[0] / (num_features * blockDim.x) + eps);
            for (int feat = 0; feat < num_features; feat++) {
                const int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
                output[idx] = input[idx] / rms;
            }
        }
    }
}

// CUDA forward function

torch::Tensor rms_norm_cuda_forward_shared(torch::Tensor input, float eps) {
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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_shared", ([&] {
        rms_norm_kernel_shared<scalar_t><<<blocks, threads_per_block, threads_per_block * sizeof(scalar_t)>>>(
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
    m.def("forward", &rms_norm_cuda_forward_shared, "RMS normalization forward with shared memory optimization (CUDA)");
}
