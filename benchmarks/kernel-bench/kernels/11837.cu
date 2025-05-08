#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel using warp-level primitives for reduction
__global__ void optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // Each thread computes its partial sum
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        // KL divergence computation: exp(log_pred) - target * log_pred
        sum += expf(log_pred) - target * log_pred;
        idx += blockDim.x * gridDim.x;
    }
    
    // Perform warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    
    // Write the warp-level result to shared memory by each warp leader
    __shared__ float warpSums[32];  // supports up to 1024 threads per block (32 warps)
    int lane = threadIdx.x & 31;    // Lane index within the warp
    int warpId = threadIdx.x >> 5;  // Warp index within the block
    if (lane == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();

    // Let thread 0 perform the final reduction across warps in this block
    if (threadIdx.x == 0) {
        float blockSum = 0.0f;
        int numWarps = (blockDim.x + 31) >> 5; // rounding up division by warp size
        for (int i = 0; i < numWarps; i++) {
            blockSum += warpSums[i];
        }
        atomicAdd(output, blockSum);
    }
}

// CUDA function exposed to PyTorch
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    optimized_kl_div_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Optimized KL divergence forward (CUDA)");
}
