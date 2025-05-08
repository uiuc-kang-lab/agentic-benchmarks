#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for KL divergence calculation 
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Get global thread ID
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warpId = tid / warpSize;
    const unsigned int lane = tid % warpSize;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    // Calculate KL divergence with grid stride loop
    #pragma unroll 4
    for (int idx = gid; idx < n; idx += blockDim.x * gridDim.x) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction first
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        partial_sums[warpId] = sum;
    }
    __syncthreads();
    
    // Block-level reduction (only first warp)
    if (warpId == 0) {
        sum = (tid < (blockDim.x / warpSize)) ? partial_sums[tid] : 0.0f;
        
        // Final warp reduction
        #pragma unroll
        for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (tid == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters optimized for occupancy
    const int threads = 256;
    const int blocks = min(65535, (n + threads - 1) / threads);
    const int shared_mem = (threads / warpSize) * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
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