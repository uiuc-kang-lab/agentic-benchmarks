#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// First stage kernel - compute partial sums per block
__global__ void kl_div_kernel_stage1(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ block_results,
    const int n) {
    
    extern __shared__ float sdata[];
    const unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 4 + tid;
    float thread_sum = 0.0f;
    
    // Process 4 elements per thread when possible
    if (i + 3 * blockDim.x < n) {
        float log_pred0 = log_predictions[i];
        float target0 = targets[i];
        thread_sum += expf(log_pred0) - target0 * log_pred0;
        
        float log_pred1 = log_predictions[i + blockDim.x];
        float target1 = targets[i + blockDim.x];
        thread_sum += expf(log_pred1) - target1 * log_pred1;
        
        float log_pred2 = log_predictions[i + 2 * blockDim.x];
        float target2 = targets[i + 2 * blockDim.x];
        thread_sum += expf(log_pred2) - target2 * log_pred2;
        
        float log_pred3 = log_predictions[i + 3 * blockDim.x];
        float target3 = targets[i + 3 * blockDim.x];
        thread_sum += expf(log_pred3) - target3 * log_pred3;
    } else {
        // Handle remaining elements
        while (i < n) {
            float log_pred = log_predictions[i];
            float target = targets[i];
            thread_sum += expf(log_pred) - target * log_pred;
            i += blockDim.x;
        }
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Reduction within shared memory
    for (int offset = blockDim.x/2; offset > 32; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    
    // Warp-level reduction
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }
    
    // Store block result
    if (tid == 0) {
        block_results[blockIdx.x] = sdata[0];
    }
}

// Second stage kernel - final reduction of block results
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
    
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[0] = sdata[0];
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Create output tensors
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);
    
    // Allocate temporary storage for block results
    auto block_results = torch::empty({blocks}, log_predictions.options());
    
    // Launch first stage
    kl_div_kernel_stage1<<<blocks, threads, threads * sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        n
    );
    
    // Launch second stage with single block
    kl_div_kernel_stage2<<<1, threads, threads * sizeof(float)>>>(
        block_results.data_ptr<float>(),
        output.data_ptr<float>(),
        blocks
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}