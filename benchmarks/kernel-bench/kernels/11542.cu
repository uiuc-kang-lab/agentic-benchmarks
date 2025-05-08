#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__global__ void hierarchical_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Thread identification
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid & 31;
    const unsigned int warp_id = tid >> 5;
    
    // Shared memory for warp results only
    __shared__ float warp_results[32];  // Max 32 warps per block
    
    // Register for local accumulation
    float sum = 0.0f;
    
    // Grid-stride loop for coalesced global memory access
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        sum += compute_kl_div(log_predictions[i], targets[i]);
    }
    
    // First level: warp-level reduction using shuffle
    sum = warp_reduce_sum(sum);
    
    // Second level: store warp results
    if (lane_id == 0) {
        warp_results[warp_id] = sum;
    }
    __syncthreads();
    
    // Third level: final reduction by first warp
    if (warp_id == 0) {
        // Load warp result if lane is within valid warp count
        float warp_sum = 0.0f;
        if (lane_id < (blockDim.x >> 5)) {
            warp_sum = warp_results[lane_id];
        }
        
        // Final warp-level reduction
        warp_sum = warp_reduce_sum(warp_sum);
        
        // Single thread atomic add to global result
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize launch configuration
    const int threads = 512;  // Increased thread count for better occupancy
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = 32 * sizeof(float);  // Only need space for warp results
    
    hierarchical_kl_div_kernel<<<blocks, threads, shared_mem>>>(
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