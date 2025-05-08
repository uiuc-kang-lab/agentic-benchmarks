#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for computing KL divergence element
__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Optimized warp reduction using warp-level primitives
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kl_div_kernel_streamed(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Shared memory for partial sums (one per warp)
    extern __shared__ float warp_sums[];
    
    // Calculate number of warps in this block
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    
    // Each thread processes multiple elements
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Coalesced memory access and loop unrolling
    #pragma unroll 4
    while (idx < n) {
        sum += compute_kl_element(log_predictions[idx], targets[idx]);
        idx += stride;
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps (done by first warp)
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
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize block size to maximize occupancy
    const int threads = 256;  // Multiple of warp size (32)
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int warps_per_block = threads / 32;
    const int shared_mem = warps_per_block * sizeof(float);

    // Create CUDA stream for asynchronous execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Launch kernel asynchronously
    kl_div_kernel_streamed<<<blocks, threads, shared_mem, stream>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    // Synchronize stream to ensure completion
    cudaStreamSynchronize(stream);
    
    // Destroy stream
    cudaStreamDestroy(stream);
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}
