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
    
    float4* log_pred_vec = (float4*)log_predictions;
    float4* target_vec = (float4*)targets;
    
    while (i + 7 * blockDim.x < n) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int vec_idx = (i + j * 4 * blockDim.x) / 4;
            float4 log_pred4 = log_pred_vec[vec_idx];
            float4 target4 = target_vec[vec_idx];
            
            thread_sum += __expf(log_pred4.x) - target4.x * log_pred4.x;
            thread_sum += __expf(log_pred4.y) - target4.y * log_pred4.y;
            thread_sum += __expf(log_pred4.z) - target4.z * log_pred4.z;
            thread_sum += __expf(log_pred4.w) - target4.w * log_pred4.w;
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
    if (num_blocks >= 4) {
        float4* block_results_vec = (float4*)block_results;
        for (int i = tid * 4; i < num_blocks - 3; i += blockDim.x * 4) {
            float4 block4 = block_results_vec[i/4];
            sum += block4.x + block4.y + block4.z + block4.w;
        }
    }
    
    for (int i = tid + ((num_blocks/4)*4); i < num_blocks; i += blockDim.x) {
        sum += block_results[i];
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}