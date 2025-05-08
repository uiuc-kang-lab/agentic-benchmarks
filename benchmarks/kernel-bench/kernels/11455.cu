#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_stride(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Calculate thread's starting position and stride
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int blockSize = blockDim.x;
    const unsigned int gridSize = blockSize * gridDim.x;
    
    // Each thread accumulates its own sum
    float sum = 0.0f;
    
    // Grid-stride loop
    unsigned int idx = bid * blockSize + tid;
    // Unroll two iterations per loop to reduce loop overhead
    while (idx + gridSize < n) {
        float log_pred1 = log_predictions[idx];
        float target1 = targets[idx];
        float log_pred2 = log_predictions[idx + gridSize];
        float target2 = targets[idx + gridSize];
        // Use fused multiply-add where appropriate
        sum += expf(log_pred1) - target1 * log_pred1;
        sum += expf(log_pred2) - target2 * log_pred2;
        idx += gridSize * 2;
    }
    // Process remaining elements
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
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
    
    // Final reduction by first warp
    if (tid < (blockSize + warpSize - 1) / warpSize) {
        sum = warp_sums[tid];
    } else {
        sum = 0.0f;
    }
    
    if (tid < warpSize) {
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
    
    // Optimize launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = min(32, (n + threadsPerBlock - 1) / threadsPerBlock);
    
    kl_div_kernel_stride<<<blocksPerGrid, threadsPerBlock>>>(
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