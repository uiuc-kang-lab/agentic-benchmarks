#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Warp size and thread identification
    const unsigned int warp_size = 32;
    const unsigned int tid = threadIdx.x;
    const unsigned int wid = threadIdx.x / warp_size;
    const unsigned int lane = threadIdx.x % warp_size;
    const unsigned int warps_per_block = blockDim.x / warp_size;
    const unsigned int global_warp_id = blockIdx.x * warps_per_block + wid;
    
    // Shared memory for partial sums - one per warp
    extern __shared__ float partial_sums[];
    float thread_sum = 0.0f;
    
    // Calculate stride for coalesced access
    const unsigned int stride = gridDim.x * blockDim.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Main computation loop with stride pattern
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
        idx += stride;
    }
    
    // Warp-level reduction first (no synchronization needed within a warp)
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        partial_sums[wid] = thread_sum;
    }
    
    __syncthreads();
    
    // Final reduction across warps (only first warp)
    if (wid == 0 && lane < warps_per_block) {
        float warp_sum = partial_sums[lane];
        
        // Warp-level reduction of the partial sums
        #pragma unroll
        for (int offset = warps_per_block/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // First thread adds to global output
        if (lane == 0) {
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
    const int threads_per_block = 256;
    const int num_warps = threads_per_block / 32;
    const int blocks = min(256, (n + threads_per_block - 1) / threads_per_block);
    const int shared_mem = num_warps * sizeof(float);
    
    kl_div_kernel<<<blocks, threads_per_block, shared_mem>>>(
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