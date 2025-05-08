#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp-level reduction using shuffle intrinsic with pragma unroll
__forceinline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_optimized(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Calculate thread indices
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int blockSize = blockDim.x;
    const unsigned int gridSize = blockSize * gridDim.x;
    
    // Each thread accumulates its own sum
    float sum = 0.0f;
    
    // Grid-stride loop with 4-way unrolling for better instruction-level parallelism
    unsigned int idx = bid * blockSize + tid;
    while (idx + 3 * gridSize < n) {
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            float log_pred = log_predictions[idx + i * gridSize];
            float target = targets[idx + i * gridSize];
            // Use fused multiply-add for better performance
            sum = fmaf(-target, log_pred, sum + expf(log_pred));
        }
        idx += gridSize * 4;
    }
    
    // Handle remaining elements
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum = fmaf(-target, log_pred, sum + expf(log_pred));
        idx += gridSize;
    }
    
    // Warp-level reduction
    sum = warpReduceSum(sum);
    
    // Block-level reduction using shared memory
    __shared__ float warp_sums[32];  // One slot per warp
    const unsigned int lane = tid % warpSize;
    const unsigned int wid = tid / warpSize;
    
    if (lane == 0) {
        warp_sums[wid] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp only
    if (tid < warpSize) {
        sum = (tid < (blockSize + warpSize - 1) / warpSize) ? warp_sums[tid] : 0.0f;
        sum = warpReduceSum(sum);
        if (lane == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize launch configuration based on occupancy
    const int threadsPerBlock = 256;
    const int blocksPerGrid = min(32, (n + threadsPerBlock - 1) / threadsPerBlock);
    
    kl_div_kernel_optimized<<<blocksPerGrid, threadsPerBlock>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA optimized)");
}