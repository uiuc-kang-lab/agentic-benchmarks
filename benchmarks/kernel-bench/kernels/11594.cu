#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute KL divergence for a single element
__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Device function for aligned vector load
__device__ __forceinline__ void load_vectors(
    const float4* src_log_pred,
    const float4* src_target,
    float* sum,
    const int idx,
    const int n) {
    
    if (idx * 4 < n) {
        float4 log_pred = src_log_pred[idx];
        float4 target = src_target[idx];
        
        *sum += compute_kl_div(log_pred.x, target.x);
        if (idx * 4 + 1 < n) *sum += compute_kl_div(log_pred.y, target.y);
        if (idx * 4 + 2 < n) *sum += compute_kl_div(log_pred.z, target.z);
        if (idx * 4 + 3 < n) *sum += compute_kl_div(log_pred.w, target.w);
    }
}

__global__ void kl_div_kernel_coalesced(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    extern __shared__ float partial_sums[];
    float sum = 0.0f;
    
    // Convert to float4 pointers for vectorized loads
    const float4* log_pred_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* target_vec = reinterpret_cast<const float4*>(targets);
    
    // Process 4 elements at a time per thread
    #pragma unroll 2
    for (int i = gid; i < (n + 3) / 4; i += blockDim.x * gridDim.x) {
        load_vectors(log_pred_vec, target_vec, &sum, i, n);
    }
    
    // Handle remaining elements
    const int remain_start = ((n + 3) / 4) * 4;
    if (gid + remain_start < n) {
        sum += compute_kl_div(
            log_predictions[gid + remain_start],
            targets[gid + remain_start]
        );
    }
    
    partial_sums[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    #pragma unroll
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward_coalesced(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Ensure alignment for float4
    const int threads = 256;
    const int blocks = min((n + threads * 4 - 1) / (threads * 4), 1024);
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel_coalesced<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_coalesced, "KL divergence forward with coalesced memory access (CUDA)");
}