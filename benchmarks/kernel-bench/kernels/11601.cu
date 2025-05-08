#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template<unsigned int blockSize>
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    extern __shared__ float sdata[];
    
    const unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 4 + tid;
    const unsigned int gridSize = blockSize * 4 * gridDim.x;
    
    float thread_sum = 0.0f;
    
    // Manual unroll of main computation loop - process 4 elements per iteration
    #pragma unroll
    while (i + 3 * blockSize < n) {
        float log_pred0 = log_predictions[i];
        float target0 = targets[i];
        thread_sum += expf(log_pred0) - target0 * log_pred0;
        
        float log_pred1 = log_predictions[i + blockSize];
        float target1 = targets[i + blockSize];
        thread_sum += expf(log_pred1) - target1 * log_pred1;
        
        float log_pred2 = log_predictions[i + 2 * blockSize];
        float target2 = targets[i + 2 * blockSize];
        thread_sum += expf(log_pred2) - target2 * log_pred2;
        
        float log_pred3 = log_predictions[i + 3 * blockSize];
        float target3 = targets[i + 3 * blockSize];
        thread_sum += expf(log_pred3) - target3 * log_pred3;
        
        i += gridSize;
    }
    
    // Handle remaining elements
    while (i < n) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        thread_sum += expf(log_pred) - target * log_pred;
        i += blockSize;
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Unrolled reduction in shared memory
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }
    
    // Final warp reduction
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    
    // Write result
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads * 4 - 1) / (threads * 4), 1024);
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel<256><<<blocks, threads, shared_mem>>>(
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