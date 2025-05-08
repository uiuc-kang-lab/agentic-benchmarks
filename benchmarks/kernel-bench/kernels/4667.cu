#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = tid / numel_per_batch;
    
    if (batch_id >= batch_size) return;
    
    const int offset_in_batch = tid % numel_per_batch;
    const int batch_offset = batch_id * num_features * numel_per_batch;
    
    __shared__ scalar_t shared_sumsq[256];
    
    // Calculate partial sum of squares
    scalar_t local_sumsq = 0.0f;
    for (int feat = 0; feat < num_features; feat++) {
        const scalar_t val = input[batch_offset + feat * numel_per_batch + offset_in_batch];
        local_sumsq += val * val;
    }
    
    // Store in shared memory
    shared_sumsq[threadIdx.x] = local_sumsq;
    __syncthreads();
    
    // Reduce within block using warp-level primitives
    if (threadIdx.x < 128) shared_sumsq[threadIdx.x] += shared_sumsq[threadIdx.x + 128];
    __syncthreads();
    if (threadIdx.x < 64) shared_sumsq[threadIdx.x] += shared_sumsq[threadIdx.x + 64];
    __syncthreads();
    
    // Final warp reduction
    if (threadIdx.x < 32) {
        scalar_t* smem = shared_sumsq;
        if (blockDim.x >= 64) smem[threadIdx.x] += smem[threadIdx.x + 32];
        smem[threadIdx.x] += smem[threadIdx.x + 16];
        smem[threadIdx.x] += smem[threadIdx.x + 8];
        smem[threadIdx.x] += smem[threadIdx.x + 4];
        smem[threadIdx.x] += smem[threadIdx.x + 2];
        smem[threadIdx.x] += smem[threadIdx.x + 1];
    }
    
    // Calculate RMS
    const scalar_t rms = sqrt(shared_sumsq[0] / num_features + eps);
    
    // Normalize
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

    const int total_threads = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward (CUDA)");
}