#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for calculating KL divergence for a single element
__device__ float compute_kl_div(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Device function for warp reduction
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void modular_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const unsigned int warp_size = 32;
    const unsigned int lane_id = threadIdx.x % warp_size;
    const unsigned int warp_id = threadIdx.x / warp_size;
    const unsigned int warps_per_block = blockDim.x / warp_size;
    
    extern __shared__ float warp_results[];
    
    float sum = 0.0f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < n; i += stride) {
        sum += compute_kl_div(log_predictions[i], targets[i]);
    }
    
    sum = warp_reduce_sum(sum);
    
    if (lane_id == 0) {
        warp_results[warp_id] = sum;
    }
    
    __syncthreads();
    
    if (warp_id == 0) {
        float warp_sum = (lane_id < warps_per_block) ? warp_results[lane_id] : 0.0f;
        warp_sum = warp_reduce_sum(warp_sum);
        
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
    
    const int threads_per_block = 256;
    const int num_warps = threads_per_block / 32;
    const int blocks = min(256, (n + threads_per_block - 1) / threads_per_block);
    const int shared_mem = num_warps * sizeof(float);
    
    modular_kl_div_kernel<<<blocks, threads_per_block, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Modular KL divergence forward (CUDA)");
}
