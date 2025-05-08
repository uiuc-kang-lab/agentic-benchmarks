#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    float local_sum = 0.0f;
    
    // Grid stride loop
    for (int idx = gid; idx < n; idx += gridDim.x * blockDim.x) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        local_sum += __expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    extern __shared__ float warp_results[];
    if (lane_id == 0) {
        warp_results[warp_id] = local_sum;
    }
    
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0 && lane_id < (blockDim.x / 32)) {
        float warp_sum = warp_results[lane_id];
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
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
    
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks = min((n + threads - 1) / threads, 1024);
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