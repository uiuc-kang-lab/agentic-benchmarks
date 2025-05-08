#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 128
#define BLOCK_ROWS 8
#define PADDING 4

__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void persistent_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    // Padded shared memory to avoid bank conflicts
    __shared__ float shared_log_pred[TILE_DIM][BLOCK_ROWS + PADDING];
    __shared__ float shared_target[TILE_DIM][BLOCK_ROWS + PADDING];
    
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    
    // Thread block processes multiple tiles before reduction
    float thread_sum = 0.0f;
    
    // Calculate starting position for this thread block
    int block_start = blockIdx.x * (TILE_DIM * BLOCK_ROWS);
    const int grid_stride = gridDim.x * (TILE_DIM * BLOCK_ROWS);
    
    // Persistent thread block processes multiple chunks
    while (block_start < n) {
        // Load multiple rows per thread
        #pragma unroll
        for (int i = 0; i < BLOCK_ROWS; i++) {
            const int row = i * TILE_DIM + tid;
            const int global_idx = block_start + row;
            
            if (global_idx < n) {
                shared_log_pred[tid][i] = log_predictions[global_idx];
                shared_target[tid][i] = targets[global_idx];
            }
        }
        __syncthreads();
        
        // Process data in shared memory
        #pragma unroll
        for (int i = 0; i < BLOCK_ROWS; i++) {
            const int row = i * TILE_DIM + tid;
            if (block_start + row < n) {
                thread_sum += compute_kl_div(
                    shared_log_pred[tid][i],
                    shared_target[tid][i]
                );
            }
        }
        __syncthreads();
        
        block_start += grid_stride;
    }
    
    // Warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);
    
    // Inter-warp reduction using shared memory
    __shared__ float warp_sums[32];
    
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        float sum = (lane_id < (blockDim.x >> 5)) ? warp_sums[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters optimized for H100
    const int threads = TILE_DIM;
    const int blocks = min((n + (TILE_DIM * BLOCK_ROWS) - 1) / (TILE_DIM * BLOCK_ROWS), 512);
    
    persistent_kl_div_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}