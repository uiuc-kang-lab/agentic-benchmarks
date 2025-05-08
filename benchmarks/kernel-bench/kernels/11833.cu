#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float compute_kldiv(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_coalesced_uniform(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    
    // Each thread processes elements with a stride of WARP_SIZE
    // This ensures coalesced memory access within each warp
    float sum = 0.0f;
    int base_idx = global_warp_id * 32 * 4 + lane_id * 4;
    const int stride = gridDim.x * warps_per_block * 32 * 4;
    
    // Main computation loop - each iteration processes 4 elements per thread
    while (base_idx < n - 3) {
        float4 log_vec = *reinterpret_cast<const float4*>(log_predictions + base_idx);
        float4 target_vec = *reinterpret_cast<const float4*>(targets + base_idx);
        
        sum += compute_kldiv(log_vec.x, target_vec.x);
        sum += compute_kldiv(log_vec.y, target_vec.y);
        sum += compute_kldiv(log_vec.z, target_vec.z);
        sum += compute_kldiv(log_vec.w, target_vec.w);
        
        base_idx += stride;
    }
    
    // Warp-level reduction
    sum = warp_reduce(sum);
    
    // Block-level reduction using shared memory
    __shared__ float warp_sums[32];
    
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // First warp reduces all warp sums
    if (warp_id == 0) {
        sum = (lane_id < warps_per_block) ? warp_sums[lane_id] : 0.0f;
        sum = warp_reduce(sum);
        
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
    
    // Handle remaining elements with a dedicated warp
    // This prevents divergence in the main computation warps
    if (blockIdx.x == 0 && warp_id == warps_per_block - 1) {
        int remaining_start = (n / 4) * 4;
        sum = 0.0f;
        
        // Each lane handles one remaining element
        for (int idx = remaining_start + lane_id; idx < n; idx += 32) {
            sum += compute_kldiv(log_predictions[idx], targets[idx]);
        }
        
        sum = warp_reduce(sum);
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward_coalesced(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;  // Must be multiple of 32
    const int warps_per_block = threads / 32;
    const int num_blocks = min((n + (threads * 4) - 1) / (threads * 4), 1024);
    
    kl_div_kernel_coalesced_uniform<<<num_blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_coalesced, "Coalesced uniform KLDiv forward (CUDA)");
}