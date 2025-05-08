#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256
#define MAX_BLOCKS 1024
#define FULL_MASK 0xffffffff

__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__global__ void optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    extern __shared__ float shared_mem[];
    float* const shared_log_pred = shared_mem;
    float* const shared_targets = &shared_mem[TILE_SIZE];
    float* const partial_sums = &shared_mem[2 * TILE_SIZE];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    
    float thread_sum = 0.0f;
    
    for (int tile_start = bid * TILE_SIZE; tile_start < n; tile_start += gridDim.x * TILE_SIZE) {
        const int remaining = n - tile_start;
        const int tile_elements = min(TILE_SIZE, remaining);
        
        if (tid < tile_elements) {
            shared_log_pred[tid] = log_predictions[tile_start + tid];
            shared_targets[tid] = targets[tile_start + tid];
        }
        __syncthreads();
        
        #pragma unroll 4
        for (int i = tid; i < tile_elements; i += blockDim.x) {
            thread_sum += compute_kl_div(shared_log_pred[i], shared_targets[i]);
        }
        __syncthreads();
    }
    
    thread_sum = warp_reduce_sum(thread_sum);
    
    if (lane_id == 0) {
        partial_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    if (warp_id == 0 && lane_id < (blockDim.x >> 5)) {
        float sum = partial_sums[lane_id];
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
    
    const int threads = TILE_SIZE;
    const int blocks = min((n + TILE_SIZE - 1) / TILE_SIZE, MAX_BLOCKS);
    const int shared_mem = (2 * TILE_SIZE + threads/32) * sizeof(float);
    
    optimized_kl_div_kernel<<<blocks, threads, shared_mem>>>(
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