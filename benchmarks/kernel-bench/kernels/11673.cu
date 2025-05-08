#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kl_div_kernel_vectorized(
    const float4* __restrict__ log_predictions,
    const float4* __restrict__ targets, 
    float* __restrict__ output,
    const int n_vec) {
    
    extern __shared__ float warp_sums[];
    
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    
    // Ensure coalesced memory access
    const int vec_idx = blockIdx.x * blockDim.x / 4 + threadIdx.x / 4;
    const int vec_stride = gridDim.x * blockDim.x / 4;
    
    float sum = 0.0f;
    
    if (vec_idx < n_vec) {
        float4 log_pred4 = log_predictions[vec_idx];
        float4 target4 = targets[vec_idx];
        
        sum += compute_kl_element(log_pred4.x, target4.x);
        sum += compute_kl_element(log_pred4.y, target4.y);
        sum += compute_kl_element(log_pred4.z, target4.z);
        sum += compute_kl_element(log_pred4.w, target4.w);
    }
    
    // Warp reduction
    sum = warp_reduce_sum(sum);
    
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0 && lane < warps_per_block) {
        float warp_sum = warp_sums[lane];
        warp_sum = warp_reduce_sum(warp_sum);
        
        if (lane == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    const int n_vec = (n + 3) / 4;
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 128;
    const int blocks = min((n_vec * 4 + threads - 1) / threads, 1024);
    const int warps_per_block = threads / 32;
    const int shared_mem = warps_per_block * sizeof(float);
    
    kl_div_kernel_vectorized<<<blocks, threads, shared_mem>>>(
        reinterpret_cast<const float4*>(log_predictions.data_ptr<float>()),
        reinterpret_cast<const float4*>(targets.data_ptr<float>()),
        output.data_ptr<float>(),
        n_vec
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}