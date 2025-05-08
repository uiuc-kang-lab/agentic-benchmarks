#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t, int FEATURES_PER_THREAD = 4>
__global__ void optimized_rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    extern __shared__ float partial_sums[];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int lane_id = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    for (int index = tid; index < batch_size * numel_per_batch; index += total_threads) {
        const int batch_id = index / numel_per_batch;
        const int offset_in_batch = index % numel_per_batch;
        const int batch_offset = batch_id * num_features * numel_per_batch;

        scalar_t sumsq = 0.0f;
        
        #pragma unroll
        for (int feat = 0; feat < num_features; feat += FEATURES_PER_THREAD) {
            scalar_t vals[FEATURES_PER_THREAD];
            #pragma unroll
            for (int j = 0; j < FEATURES_PER_THREAD && (feat + j) < num_features; j++) {
                vals[j] = input[batch_offset + (feat + j) * numel_per_batch + offset_in_batch];
                sumsq += vals[j] * vals[j];
            }
        }

        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
        }

        if (lane_id == 0) {
            partial_sums[warp_id] = sumsq;
        }
        __syncthreads();

        if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
            sumsq = partial_sums[lane_id];
            #pragma unroll
            for (int offset = (blockDim.x/warpSize)/2; offset > 0; offset /= 2) {
                sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
            }
        }

        if (warp_id == 0 && lane_id == 0) {
            partial_sums[0] = sumsq;
        }
        __syncthreads();
        
        const scalar_t rms = sqrt(partial_sums[0] / num_features + eps);

        #pragma unroll
        for (int feat = 0; feat < num_features; feat += FEATURES_PER_THREAD) {
            #pragma unroll
            for (int j = 0; j < FEATURES_PER_THREAD && (feat + j) < num_features; j++) {
                const int idx = batch_offset + (feat + j) * numel_per_batch + offset_in_batch;
                output[idx] = input[idx] / rms;
            }
        }
    }
}

torch::Tensor rms_norm_cuda_forward_optimized(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    int numel_per_batch = 1;
    for(int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    const int threads_per_block = 256;
    const int blocks = (batch_size * numel_per_batch + threads_per_block - 1) / threads_per_block;
    const int shared_mem_size = (threads_per_block / 32) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_optimized", ([&] {
        optimized_rms_norm_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
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