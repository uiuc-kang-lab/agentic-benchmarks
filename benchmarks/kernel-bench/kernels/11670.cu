#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for computing KL divergence element
__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Warp reduction using shuffle operations
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// CUDA kernel with warp shuffle reduction
__global__ void kl_div_kernel_shuffle(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid % 32;
    const unsigned int warp_id = tid / 32;
    int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    // Compute local sum
    float sum = 0.0f;
    while (idx < n) {
        sum += compute_kl_element(log_predictions[idx], targets[idx]);
        idx += stride;
    }
    
    // Warp-level reduction using shuffle
    sum = warp_reduce_sum(sum);
    
    // First thread in each warp writes to shared memory
    __shared__ float warp_sums[8]; // Assuming block size = 256
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps (done by first warp)
    if (warp_id == 0 && lane_id < (blockDim.x / 32)) {
        float warp_sum = warp_sums[lane_id];
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
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    
    kl_div_kernel_shuffle<<<blocks, threads>>>(
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