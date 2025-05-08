#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel_warp_shuffle(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid % warpSize;
    const unsigned int wid = tid / warpSize;
    const unsigned int idx = blockIdx.x * blockDim.x + tid;
    
    // Shared memory for warp results only
    __shared__ float warp_results[32];  // Max 32 warps per block
    
    float sum = 0.0f;
    
    // Compute KL divergence with coalesced memory access
    #pragma unroll 4
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        warp_results[wid] = sum;
    }
    
    __syncthreads();
    
    // Final reduction across warps using the first warp
    if (wid == 0) {
        sum = (lane_id < (blockDim.x / warpSize)) ? warp_results[lane_id] : 0.0f;
        
        // Warp-level reduction for final results
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // Only the first thread in the block writes to global memory
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
    
    // Optimize launch parameters
    const int threads = 256;  // Must be multiple of warpSize (32)
    const int max_blocks = 128;
    const int blocks = min((n + threads - 1) / threads, max_blocks);
    
    kl_div_kernel_warp_shuffle<<<blocks, threads>>>(
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