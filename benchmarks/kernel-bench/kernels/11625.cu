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
    const unsigned int stride = blockDim.x * gridDim.x;
    
    float thread_sum = 0.0f;
    float4 log_pred_vec, target_vec;
    
    while (i + 7 * blockDim.x < n) {
        #pragma unroll
        for (int j = 0; j < 8; j += 4) {
            int idx = i + j * blockDim.x;
            if (idx + 3 * blockDim.x < n) {
                log_pred_vec = *reinterpret_cast<const float4*>(&log_predictions[idx]);
                target_vec = *reinterpret_cast<const float4*>(&targets[idx]);
                
                thread_sum += __expf(log_pred_vec.x) - target_vec.x * log_pred_vec.x;
                thread_sum += __expf(log_pred_vec.y) - target_vec.y * log_pred_vec.y;
                thread_sum += __expf(log_pred_vec.z) - target_vec.z * log_pred_vec.z;
                thread_sum += __expf(log_pred_vec.w) - target_vec.w * log_pred_vec.w;
            }
        }
        i += stride * 8;
    }
    
    while (i < n) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        thread_sum += __expf(log_pred) - target * log_pred;
        i += stride;
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();
    
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    if (tid < 32) warpReduce<256>(sdata, tid);
    
    if (tid == 0) block_results[blockIdx.x] = sdata[0];
}

__global__ void kl_div_kernel_stage2(
    const float* __restrict__ block_results,
    float* __restrict__ output,
    const int num_blocks,
    const float normalizer) {
    
    extern __shared__ float sdata[];
    const unsigned int tid = threadIdx.x;
    
    float sum = 0.0f;
    int i = tid;
    while (i < num_blocks - 3) {
        float4 block_vec = *reinterpret_cast<const float4*>(&block_results[i]);
        sum += block_vec.x + block_vec.y + block_vec.z + block_vec.w;
        i += blockDim.x;
    }
    
    while (i < num_blocks) {
        sum += block_results[i];
        i += blockDim.x;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    if (tid < 32) warpReduce<256>(sdata, tid);
    
    if (tid == 0) {
        output[0] = sdata[0] * normalizer;
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads * 8 - 1) / (threads * 8), 1024);
    const float normalizer = 1.0f / static_cast<float>(n);
    
    auto block_results = torch::empty({blocks}, log_predictions.options());
    
    kl_div_kernel_stage1<<<blocks, threads, threads * sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        n
    );
    
    kl_div_kernel_stage2<<<1, threads, threads * sizeof(float)>>>(
        block_results.data_ptr<float>(),
        output.data_ptr<float>(),
        blocks,
        normalizer
    );
    
    return output;
}