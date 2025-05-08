#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    // Shared memory for inter-warp reduction
    __shared__ float warp_results[NUM_WARPS];
    
    // Calculate thread local sum using grid stride loop
    float sum = 0.0f;
    for (int idx = blockIdx.x * BLOCK_SIZE + tid; 
         idx < n; 
         idx += gridDim.x * BLOCK_SIZE) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp participates)
    if (warp_id == 0) {
        float warp_sum = (tid < NUM_WARPS) ? warp_results[tid] : 0.0f;
        
        // Warp-level reduction of the final results
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        if (tid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    constexpr int BLOCK_SIZE = 256;
    const int num_sms = 108; // H100 has 108 SMs
    const int blocks_per_sm = 2;
    const int num_blocks = std::min(
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
        num_sms * blocks_per_sm
    );
    
    kl_div_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
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