#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#define MAX_FEATURES_CONST 1024
#define BLOCK_SIZE 256
#define WARP_SIZE 32

__constant__ int d_offsets[MAX_FEATURES_CONST];

template <typename scalar_t>
__global__ void optimized_rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    __shared__ scalar_t warp_sums[BLOCK_SIZE / WARP_SIZE];

    for (int index = tid; index < batch_size * numel_per_batch; index += total_threads) {
        const int batch_id = index / numel_per_batch;
        const int offset_in_batch = index % numel_per_batch;
        const int batch_offset = batch_id * num_features * numel_per_batch;

        scalar_t thread_sum = 0.0f;
        
        #pragma unroll 4
        for (int feat = 0; feat < num_features; feat++) {
            const int pos = batch_offset + d_offsets[feat] + offset_in_batch;
            const scalar_t val = input[pos];
            thread_sum += val * val;
        }

        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }

        if (lane_id == 0) {
            warp_sums[warp_id] = thread_sum;
        }
        
        __syncthreads();

        if (warp_id == 0 && lane_id < (blockDim.x / WARP_SIZE)) {
            scalar_t warp_sum = warp_sums[lane_id];
            
            #pragma unroll
            for (int offset = (BLOCK_SIZE/WARP_SIZE)/2; offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
            }

            if (lane_id == 0) {
                warp_sums[0] = warp_sum;
            }
        }

        __syncthreads();

        const scalar_t rms = sqrt(warp_sums[0] / num_features + eps);

        #pragma unroll 4
        for (int feat = 0; feat < num_features; feat++) {
            const int pos = batch_offset + d_offsets[feat] + offset_in_batch;
            output[pos] = input[pos] / rms;
        }
    }
}

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }
    
    if (num_features > MAX_FEATURES_CONST) {
        throw std::runtime_error("num_features exceeds constant memory limit");
    }
    
    std::vector<int> offsets(num_features);
    for (int i = 0; i < num_features; i++) {
        offsets[i] = i * numel_per_batch;
    }
    
    cudaMemcpyToSymbol(d_offsets, offsets.data(), num_features * sizeof(int));
    
    const int total_threads = batch_size * numel_per_batch;
    const int blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        optimized_rms_norm_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(
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