#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <limits>

template <typename scalar_t, int BLOCK_SIZE>
__global__ void adaptive_argmin_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size,
    bool use_cooperative_groups) {

    const int tid = threadIdx.x;
    const int slice_idx = blockIdx.x;
    if (slice_idx >= outer_size * inner_size) return;

    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;

    // Use cooperative groups for large K values
    if (use_cooperative_groups) {
        auto block = cooperative_groups::this_thread_block();
        
        scalar_t local_min = std::numeric_limits<scalar_t>::max();
        int local_min_idx = 0;

        // Vectorized loading for coalesced memory access
        using vec4_t = typename cuda_vec4<scalar_t>::type;
        const vec4_t* x_vec = reinterpret_cast<const vec4_t*>(&x[outer * (K * inner_size) + inner]);
        
        #pragma unroll 4
        for (int k = tid; k < K/4; k += BLOCK_SIZE) {
            vec4_t vals = __ldg(&x_vec[k * inner_size]);
            scalar_t* val_arr = reinterpret_cast<scalar_t*>(&vals);
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                if (val_arr[i] < local_min) {
                    local_min = val_arr[i];
                    local_min_idx = k*4 + i;
                }
            }
        }

        // Handle remaining elements
        for (int k = (K/4)*4 + tid; k < K; k += BLOCK_SIZE) {
            scalar_t val = __ldg(&x[outer * (K * inner_size) + k * inner_size + inner]);
            if (val < local_min) {
                local_min = val;
                local_min_idx = k;
            }
        }

        __shared__ scalar_t s_min_vals[BLOCK_SIZE];
        __shared__ int s_min_inds[BLOCK_SIZE];

        s_min_vals[tid] = local_min;
        s_min_inds[tid] = local_min_idx;
        block.sync();

        // Warp-level reduction first
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            scalar_t other_val = __shfl_down_sync(0xffffffff, local_min, offset);
            int other_idx = __shfl_down_sync(0xffffffff, local_min_idx, offset);
            if (other_val < local_min) {
                local_min = other_val;
                local_min_idx = other_idx;
            }
        }

        // Block-level reduction
        if (tid % warpSize == 0) {
            s_min_vals[tid/warpSize] = local_min;
            s_min_inds[tid/warpSize] = local_min_idx;
        }
        block.sync();

        // Final reduction by first thread
        if (tid == 0) {
            local_min = s_min_vals[0];
            local_min_idx = s_min_inds[0];
            for (int i = 1; i < BLOCK_SIZE/warpSize; i++) {
                if (s_min_vals[i] < local_min) {
                    local_min = s_min_vals[i];
                    local_min_idx = s_min_inds[i];
                }
            }
            output[slice_idx] = local_min_idx;
        }
    } else {
        // Original implementation for small K values
        scalar_t local_min = std::numeric_limits<scalar_t>::max();
        int local_min_idx = 0;

        for (int k = tid; k < K; k += BLOCK_SIZE) {
            scalar_t val = __ldg(&x[outer * (K * inner_size) + k * inner_size + inner]);
            if (val < local_min) {
                local_min = val;
                local_min_idx = k;
            }
        }

        __shared__ scalar_t s_min_vals[BLOCK_SIZE];
        __shared__ int s_min_inds[BLOCK_SIZE];

        s_min_vals[tid] = local_min;
        s_min_inds[tid] = local_min_idx;
        __syncthreads();

        for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (s_min_vals[tid + stride] < s_min_vals[tid]) {
                    s_min_vals[tid] = s_min_vals[tid + stride];
                    s_min_inds[tid] = s_min_inds[tid + stride];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[slice_idx] = s_min_inds[0];
        }
    }
}