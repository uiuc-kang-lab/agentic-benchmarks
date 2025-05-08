#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

template <typename scalar_t, bool USE_2D_GRID>
__global__ void adaptive_log_softmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    int batch_idx = blockIdx.x;
    const scalar_t* row = input + batch_idx * dim_size;
    scalar_t* out_row = output + batch_idx * dim_size;

    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    scalar_t local_max, local_sum;
    int tid, total_threads;

    if (USE_2D_GRID) {
        tid = threadIdx.y * blockDim.x + threadIdx.x;
        total_threads = blockDim.x * blockDim.y;
        
        local_max = -std::numeric_limits<scalar_t>::infinity();
        for (int j = tid; j < dim_size; j += total_threads) {
            local_max = max(local_max, row[j]);
        }

        unsigned int mask = 0xffffffff;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            local_max = max(local_max, __shfl_down_sync(mask, local_max, offset));
        }

        if (threadIdx.x == 0) {
            sdata[threadIdx.y] = local_max;
        }
    } else {
        tid = threadIdx.x;
        total_threads = blockDim.x;
        
        local_max = -std::numeric_limits<scalar_t>::infinity();
        for (int idx = tid; idx < dim_size; idx += total_threads) {
            local_max = max(local_max, row[idx]);
        }
        sdata[tid] = local_max;
    }
    __syncthreads();

    if (tid == 0) {
        scalar_t max_val = sdata[0];
        for (int i = 1; i < (USE_2D_GRID ? blockDim.y : blockDim.x); i++) {
            max_val = max(max_val, sdata[i]);
        }
        sdata[0] = max_val;
    }
    __syncthreads();
    
    scalar_t max_val = sdata[0];

    local_sum = 0;
    for (int idx = tid; idx < dim_size; idx += total_threads) {
        local_sum += exp(row[idx] - max_val);
    }

    if (USE_2D_GRID) {
        unsigned int mask = 0xffffffff;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
        }
        if (threadIdx.x == 0) {
            sdata[threadIdx.y] = local_sum;
        }
    } else {
        sdata[tid] = local_sum;
    }
    __syncthreads();

    if (tid == 0) {
        scalar_t sum = 0;
        for (int i = 0; i < (USE_2D_GRID ? blockDim.y : blockDim.x); i++) {
            sum += sdata[i];
        }
        sdata[0] = log(sum);
    }
    __syncthreads();

    scalar_t log_sum = sdata[0];

    for (int idx = tid; idx < dim_size; idx += total_threads) {
        out_row[idx] = (row[idx] - max_val) - log_sum;
    }
}