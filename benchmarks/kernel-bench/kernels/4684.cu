#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int d_batch_size;
__constant__ int d_num_features;
__constant__ int d_numel_per_batch;
__constant__ float d_eps;

template <typename scalar_t>
__device__ scalar_t calculate_sumsq(
    const scalar_t* __restrict__ input,
    const int batch_offset,
    const int offset_in_batch
) {
    scalar_t sumsq = 0.0f;
    #pragma unroll 4
    for (int feat = 0; feat < d_num_features; feat++) {
        const scalar_t val = input[batch_offset + feat * d_numel_per_batch + offset_in_batch];
        sumsq += val * val;
    }
    return sumsq;
}

template <typename scalar_t>
__device__ void normalize_features(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_offset,
    const int offset_in_batch,
    const scalar_t inv_rms
) {
    #pragma unroll 4
    for (int feat = 0; feat < d_num_features; feat++) {
        const int idx = batch_offset + feat * d_numel_per_batch + offset_in_batch;
        output[idx] = input[idx] * inv_rms;
    }
}

template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = tid / d_numel_per_batch;
    
    if (batch_id >= d_batch_size) return;
    
    const int offset_in_batch = tid % d_numel_per_batch;
    const int batch_offset = batch_id * d_num_features * d_numel_per_batch;

    // Calculate sum of squares
    const scalar_t sumsq = calculate_sumsq(
        input, batch_offset, offset_in_batch
    );
    
    // Calculate inverse RMS for multiplication instead of division
    const scalar_t inv_rms = rsqrt(sumsq / d_num_features + d_eps);
    
    // Normalize
    normalize_features(
        input, output, batch_offset, offset_in_batch, inv_rms
    );
}

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    int numel_per_batch = 1;
    for(int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Copy constants to device constant memory
    cudaMemcpyToSymbol(d_batch_size, &batch_size, sizeof(int));
    cudaMemcpyToSymbol(d_num_features, &num_features, sizeof(int));
    cudaMemcpyToSymbol(d_numel_per_batch, &numel_per_batch, sizeof(int));
    cudaMemcpyToSymbol(d_eps, &eps, sizeof(float));

    const int total_threads = batch_size * numel_per_batch;
    const int threads_per_block = 512;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward (CUDA)");
}