#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ block_sumsq,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    extern __shared__ scalar_t shared_sumsq[];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = tid / numel_per_batch;
    
    if (batch_id >= batch_size) return;
    
    const int offset_in_batch = tid % numel_per_batch;
    const int batch_offset = batch_id * num_features * numel_per_batch;
    
    // Initialize shared memory
    shared_sumsq[threadIdx.x] = 0.0f;
    
    // Calculate local sum of squares
    for (int feat = 0; feat < num_features; feat++) {
        const scalar_t val = input[batch_offset + feat * numel_per_batch + offset_in_batch];
        shared_sumsq[threadIdx.x] += val * val;
    }
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sumsq[threadIdx.x] += shared_sumsq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // One thread per block atomically adds to global sum
    if (threadIdx.x == 0) {
        atomicAdd(&block_sumsq[batch_id], shared_sumsq[0]);
    }
    __syncthreads();
    
    // Calculate RMS and normalize
    const scalar_t rms = sqrt(block_sumsq[batch_id] / (num_features * numel_per_batch) + eps);
    
    for (int feat = 0; feat < num_features; feat++) {
        const int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
        output[idx] = input[idx] / rms;
    }
}

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    int numel_per_batch = 1;
    for(int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    const int threads_per_block = 256;
    const int total_threads = batch_size * numel_per_batch;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    // Allocate temporary storage for block sums
    auto block_sumsq = torch::zeros({batch_size}, input.options());
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<blocks, threads_per_block, threads_per_block * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_sumsq.data_ptr<scalar_t>(),
            batch_size,
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