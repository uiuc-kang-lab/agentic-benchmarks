#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp-level reduction using shuffle intrinsic with pragma unroll
__inline__ __device__ float warpReduceSum(float val) {
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
    
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int blockSize = blockDim.x;
    const unsigned int gridSize = blockSize * gridDim.x;
    
    // Use registers for local accumulation
    float sum = 0.0f;
    
    // Grid-stride loop with 4-way unrolling
    unsigned int idx = bid * blockSize + tid;
    while (idx + 3 * gridSize < n) {
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            const unsigned int current_idx = idx + i * gridSize;
            const float log_pred = __ldg(&log_predictions[current_idx];
            const float target = targets[current_idx];
            // Use fused multiply-add (FMA) instruction
            sum += __expf(log_pred) - target * log_pred;
        }
        idx += gridSize * 4;
    }
    
    // Handle remaining elements
    while (idx < n) {
        const float log_pred = __ldg(&log_predictions[idx];
        const float target = targets[idx];
        sum += __expf(log_pred) - target * log_pred;
        idx += gridSize;
    }
    
    // Warp-level reduction
    sum = warpReduceSum(sum);
    
    // Block-level reduction using minimal shared memory
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
    
    // Optimize launch configuration based on SM count
    const int threadsPerBlock = 256;
    const int maxBlocks = 32;  // Limit blocks to reduce atomic contention
    const int blocksPerGrid = min(maxBlocks, (n + threadsPerBlock - 1) / threadsPerBlock);
    
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