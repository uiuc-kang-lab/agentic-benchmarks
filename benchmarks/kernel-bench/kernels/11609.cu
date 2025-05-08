#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<unsigned int blockSize>
__device__ __forceinline__ void warpReduce(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void kl_div_kernel_stage1(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ block_results,
    const int n) {
    
    extern __shared__ float sdata[];
    const unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 8 + tid;
    const unsigned int gridSize = blockDim.x * 8 * gridDim.x;
    
    float thread_sum = 0.0f;
    
    while (i + 7 * blockDim.x < n) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float log_pred = log_predictions[i + j * blockDim.x];
            float target = targets[i + j * blockDim.x];
            thread_sum += __expf(log_pred) - target * log_pred;
        }
        i += gridSize;
    }
    
    while (i < n) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        thread_sum += __expf(log_pred) - target * log_pred;
        i += blockDim.x;
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();
    
    if (blockDim.x >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }
    
    if (tid < 32) warpReduce<256>(sdata, tid);
    
    if (tid == 0) {
        block_results[blockIdx.x] = sdata[0];
    }
}

__global__ void kl_div_kernel_stage2(
    const float* __restrict__ block_results,
    float* __restrict__ output,
    const int num_blocks) {
    
    extern __shared__ float sdata[];
    const unsigned int tid = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += block_results[i];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) {
        output[0] = sdata[0];
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads * 8 - 1) / (threads * 8), 1024);
    
    auto block_results = torch::empty({blocks}, log_predictions.options());
    
    kl_div_kernel_stage1<<<blocks, threads, threads * sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        n
    );
    
    kl_div_kernel_stage2<<<1, 256, 256 * sizeof(float)>>>(
        block_results.data_ptr<float>(),
        output.data_ptr<float>(),
        blocks
    );
    
    return output / static_cast<float>(n);
}