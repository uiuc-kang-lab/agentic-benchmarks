#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid % 32;
    const unsigned int warp_id = tid / 32;
    const unsigned int warps_per_block = blockDim.x / 32;
    
    // Shared memory only needed for warp results
    extern __shared__ float warp_results[];
    
    float sum = 0.0f;
    
    // Process elements with grid stride, maintaining coalesced access
    for (int idx = blockIdx.x * blockDim.x + tid; idx < n; idx += blockDim.x * gridDim.x) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // First level reduction: warp-level using shuffle
    sum = warp_reduce(sum);
    
    // Store the warp result
    if (lane_id == 0) {
        warp_results[warp_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction: only first warp reduces across all warps
    if (warp_id == 0 && lane_id < warps_per_block) {
        float warp_sum = warp_results[lane_id];
        
        // Final warp shuffle reduction
        warp_sum = warp_reduce(warp_sum);
        
        // First thread adds to global output
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters optimized for warp-based processing
    const int threads = 256; // 8 warps per block
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int warps_per_block = threads / 32;
    const int shared_mem = warps_per_block * sizeof(float);
    
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