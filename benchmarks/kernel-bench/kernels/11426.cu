#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__
float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void optimized_kl_div_kernel_v3(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int grid_stride = gridDim.x * blockDim.x;
    
    // Use vectorized loads when possible
    const float4* log_pred_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targets_vec = reinterpret_cast<const float4*>(targets);
    
    // Shared memory for partial sums (one per warp)
    extern __shared__ float warp_sums[];
    
    float sum = 0.0f;
    
    // Calculate starting index for this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process elements in chunks of 4 using vectorized loads
    int vec_idx = idx / 4;
    const int vec_stride = grid_stride / 4;
    
    while (vec_idx * 4 < n) {
        if (vec_idx * 4 + 3 < n) {
            float4 log_pred = log_pred_vec[vec_idx];
            float4 target = targets_vec[vec_idx];
            
            sum += expf(log_pred.x) - target.x * log_pred.x;
            sum += expf(log_pred.y) - target.y * log_pred.y;
            sum += expf(log_pred.z) - target.z * log_pred.z;
            sum += expf(log_pred.w) - target.w * log_pred.w;
        } else {
            // Handle remaining elements individually
            for (int i = vec_idx * 4; i < n && i < vec_idx * 4 + 4; i++) {
                float log_pred = log_predictions[i];
                float target = targets[i];
                sum += expf(log_pred) - target * log_pred;
            }
        }
        vec_idx += vec_stride;
    }
    
    // Warp-level reduction using intrinsics
    sum = warp_reduce_sum(sum);
    
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction across warps (done by first warp)
    if (warp_id == 0) {
        sum = (lane_id < warps_per_block) ? warp_sums[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Optimize thread/block configuration
    const int threads_per_block = 512; // Increased for better occupancy
    const int max_blocks = 1024; // Increased for better parallelism
    const int num_blocks = min(max_blocks, (n + threads_per_block - 1) / threads_per_block);
    
    // Shared memory size (one float per warp)
    const int warps_per_block = threads_per_block / 32;
    const int shared_mem = warps_per_block * sizeof(float);
    
    auto output = torch::zeros({1}, log_predictions.options());
    
    optimized_kl_div_kernel_v3<<<num_blocks, threads_per_block, shared_mem>>>(
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