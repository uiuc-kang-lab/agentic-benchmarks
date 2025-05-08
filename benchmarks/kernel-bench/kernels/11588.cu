#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute KL divergence for a single element
__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Device function for warp-level reduction without synchronization
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_minimal_sync(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    // Shared memory for partial sums (one element per warp)
    extern __shared__ float warp_sums[];
    
    // Compute local sum without synchronization
    float sum = 0.0f;
    #pragma unroll 4
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        sum += compute_kl_div(log_predictions[i], targets[i]);
    }
    
    // First level reduction within warps (no sync needed)
    sum = warp_reduce_sum(sum);
    
    // Store warp results in shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    
    // Single sync point needed before final reduction
    __syncthreads();
    
    // Final reduction across warps (performed by first warp)
    if (warp_id == 0 && lane_id < (blockDim.x >> 5)) {
        float warp_sum = warp_sums[lane_id];
        warp_sum = warp_reduce_sum(warp_sum);
        
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward_minimal_sync(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int warps_per_block = (threads + 31) >> 5;
    const int shared_mem = warps_per_block * sizeof(float);
    
    kl_div_kernel_minimal_sync<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_minimal_sync, "KL divergence forward with minimal synchronization (CUDA)");
}