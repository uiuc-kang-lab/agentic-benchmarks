#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 1024
#define BLOCK_SIZE 256

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_stage1(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ block_results,
    const int n) {
    
    __shared__ float shared_log_pred[TILE_SIZE];
    __shared__ float shared_targets[TILE_SIZE];
    __shared__ float partial_sums[BLOCK_SIZE];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    
    float thread_sum = 0.0f;
    
    // Process tiles
    for (unsigned int tile = bid; tile < num_tiles; tile += gridDim.x) {
        const unsigned int tile_start = tile * TILE_SIZE;
        const unsigned int tile_elements = min(TILE_SIZE, n - tile_start);
        
        // Load tile into shared memory
        #pragma unroll 4
        for (unsigned int i = tid; i < tile_elements; i += BLOCK_SIZE) {
            shared_log_pred[i] = log_predictions[tile_start + i];
            shared_targets[i] = targets[tile_start + i];
        }
        __syncthreads();
        
        // Process elements in shared memory
        #pragma unroll 4
        for (unsigned int i = tid; i < tile_elements; i += BLOCK_SIZE) {
            const float log_pred = shared_log_pred[i];
            const float target = shared_targets[i];
            thread_sum += __expf(log_pred) - target * log_pred;
        }
        __syncthreads();
    }
    
    // Store partial sum
    partial_sums[tid] = thread_sum;
    __syncthreads();
    
    // Reduce within block
    if (tid < 128) partial_sums[tid] += partial_sums[tid + 128];
    __syncthreads();
    if (tid < 64) partial_sums[tid] += partial_sums[tid + 64];
    __syncthreads();
    
    // Final warp reduction
    if (tid < 32) {
        float warp_sum = partial_sums[tid];
        warp_sum += partial_sums[tid + 32];
        warp_sum = warp_reduce(warp_sum);
        
        if (tid == 0) {
            block_results[bid] = warp_sum;
        }
    }
}

__global__ void kl_div_kernel_stage2(
    const float* __restrict__ block_results,
    float* __restrict__ output,
    const int num_blocks,
    const float normalizer) {
    
    __shared__ float shared_data[BLOCK_SIZE];
    const unsigned int tid = threadIdx.x;
    
    float thread_sum = 0.0f;
    
    // Load and sum block results
    for (int i = tid; i < num_blocks; i += BLOCK_SIZE) {
        thread_sum += block_results[i];
    }
    
    shared_data[tid] = thread_sum;
    __syncthreads();
    
    // Reduce within block
    if (tid < 128) shared_data[tid] += shared_data[tid + 128];
    __syncthreads();
    if (tid < 64) shared_data[tid] += shared_data[tid + 64];
    __syncthreads();
    
    // Final warp reduction
    if (tid < 32) {
        float warp_sum = shared_data[tid];
        if (tid < 32) warp_sum += shared_data[tid + 32];
        warp_sum = warp_reduce(warp_sum);
        
        if (tid == 0) {
            output[0] = warp_sum * normalizer;
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int blocks = min((n + TILE_SIZE - 1) / TILE_SIZE, 1024);
    const float normalizer = 1.0f / static_cast<float>(n);
    
    auto block_results = torch::empty({blocks}, log_predictions.options());
    
    kl_div_kernel_stage1<<<blocks, BLOCK_SIZE>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        n
    );
    
    kl_div_kernel_stage2<<<1, BLOCK_SIZE>>>(
        block_results.data_ptr<float>(),
        output.data_ptr<float>(),
        blocks,
        normalizer
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}