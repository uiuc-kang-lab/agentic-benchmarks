#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

template <typename scalar_t, int BLOCK_SIZE>
__global__ void adaptive_log_softmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size,
    bool use_2d_grid) {

    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;
    
    if (!use_2d_grid) {
        __shared__ scalar_t sdata[BLOCK_SIZE];
        
        scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
        for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
            local_max = max(local_max, input_row[idx]);
        }
        sdata[threadIdx.x] = local_max;
        __syncthreads();

        #pragma unroll
        for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        scalar_t max_val = sdata[0];
        __syncthreads();

        scalar_t local_sum = 0;
        for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
            local_sum += exp(input_row[idx] - max_val);
        }
        sdata[threadIdx.x] = local_sum;
        __syncthreads();

        #pragma unroll
        for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            }
            __syncthreads();
        }
        scalar_t sum = sdata[0];
        scalar_t log_sum = log(sum);

        for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
            output_row[idx] = (input_row[idx] - max_val) - log_sum;
        }
    } else {
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int total_threads = blockDim.x * blockDim.y;
        
        scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
        for (int j = tid; j < dim_size; j += total_threads) {
            local_max = max(local_max, input_row[j]);
        }

        unsigned int mask = 0xffffffff;
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            scalar_t other = __shfl_down_sync(mask, local_max, offset);
            local_max = max(local_max, other);
        }

        extern __shared__ char smem[];
        scalar_t* smax = reinterpret_cast<scalar_t*>(smem);
        int warp_id = threadIdx.y;
        
        if (threadIdx.x == 0) smax[warp_id] = local_max;
        __syncthreads();

        __shared__ scalar_t max_val;
        if (tid == 0) {
            max_val = smax[0];
            for (int i = 1; i < blockDim.y; i++) {
                max_val = max(max_val, smax[i]);
            }
        }
        __syncthreads();

        scalar_t local_sum = 0;
        for (int j = tid; j < dim_size; j += total_threads) {
            local_sum += exp(input_row[j] - max_val);
        }
        
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
        }

        scalar_t* ssum = smax + blockDim.y;
        if (threadIdx.x == 0) ssum[warp_id] = local_sum;
        __syncthreads();

        __shared__ scalar_t sum;
        if (tid == 0) {
            sum = 0;
            for (int i = 0; i < blockDim.y; i++) {
                sum += ssum[i];
            }
        }
        __syncthreads();

        scalar_t log_sum = log(sum);
        
        for (int j = tid; j < dim_size; j += total_threads) {
            output_row[j] = (input_row[j] - max_val) - log_sum;
        }
    }
}