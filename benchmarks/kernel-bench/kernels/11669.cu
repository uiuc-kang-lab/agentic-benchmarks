#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for computing KL divergence element
__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Device function for warp reduction
__device__ __forceinline__ void warp_reduce(volatile float* sdata, unsigned int tid) {
    if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
    if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
    if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
    if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
    if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
    if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void kl_div_kernel_shared(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int blockSize = blockDim.x;
    
    // Shared memory declarations
    extern __shared__ float shared_mem[];
    float* shared_log_pred = shared_mem;
    float* shared_targets = &shared_mem[blockSize];
    float* partial_sums = &shared_mem[2 * blockSize];
    
    float sum = 0.0f;
    
    // Process data in tiles
    const int tiles = (n + blockSize - 1) / blockSize;
    for(int tile = 0; tile < tiles; tile++) {
        const int idx = tile * blockSize + tid;
        
        // Load data into shared memory
        if(idx < n) {
            shared_log_pred[tid] = log_predictions[idx];
            shared_targets[tid] = targets[idx];
        }
        __syncthreads();
        
        // Process data from shared memory
        if(idx < n) {
            sum += compute_kl_element(shared_log_pred[tid], shared_targets[tid]);
        }
        __syncthreads();
    }
    
    // Store partial sum
    partial_sums[tid] = sum;
    __syncthreads();
    
    // Reduction within block
    for(int stride = blockSize/2; stride > 32; stride >>= 1) {
        if(tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    // Final warp reduction
    if(tid < 32) {
        warp_reduce(partial_sums, tid);
    }
    
    // Write result
    if(tid == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    
    // Allocate shared memory for log_predictions, targets, and partial sums
    const int shared_mem = 3 * threads * sizeof(float);
    
    kl_div_kernel_shared<<<blocks, threads, shared_mem>>>(
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