#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__
float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_preds,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int n) {
    
    // Vectorized processing (4 elements per thread)
    constexpr int VEC_SIZE = 4;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_tid = tid * VEC_SIZE;
    
    float sum = 0.0f;
    
    // Process vectorized elements
    if (vec_tid + VEC_SIZE <= n) {
        float4 log_preds_vec = *reinterpret_cast<const float4*>(log_preds + vec_tid);
        float4 targets_vec = *reinterpret_cast<const float4*>(targets + vec_tid);
        
        sum += expf(log_preds_vec.x) - targets_vec.x * log_preds_vec.x;
        sum += expf(log_preds_vec.y) - targets_vec.y * log_preds_vec.y;
        sum += expf(log_preds_vec.z) - targets_vec.z * log_preds_vec.z;
        sum += expf(log_preds_vec.w) - targets_vec.w * log_preds_vec.w;
    }
    // Handle remaining elements
    else {
        for (int i = vec_tid; i < min(vec_tid + VEC_SIZE, n); ++i) {
            float log_pred = log_preds[i];
            float target = targets[i];
            sum += expf(log_pred) - target * log_pred;
        }
    }

    // Warp reduction
    sum = warp_reduce(sum);
    
    // Cross-warp reduction
    __shared__ float warp_sums[32];
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction
    if (warp_id == 0) {
        sum = lane_id < (blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
        sum = warp_reduce(sum);
        
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
    
    // Optimized launch config
    constexpr int VEC_SIZE = 4;
    const int threads = 256;
    const int blocks = (n + threads * VEC_SIZE - 1) / (threads * VEC_SIZE);
    
    kl_div_kernel<<<blocks, threads, 32 * sizeof(float)>>>(
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