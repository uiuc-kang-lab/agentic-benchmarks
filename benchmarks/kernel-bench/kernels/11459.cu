#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle intrinsic
__inline__ __device__ float warpReduceSum(float val) {
    // Use full mask 0xffffffff
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Combined CUDA kernel for KL divergence calculation that uses grid-stride loops and warp-level reduction
__global__ void kl_div_kernel_combined(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {

    float local_sum = 0.0f;
    // Grid-stride loop to cover all elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        float lp = log_predictions[idx];
        float t  = targets[idx];
        // F.kl_div: exp(log_predictions) - targets * log_predictions
        local_sum += expf(lp) - t * lp;
    }

    // Intra-warp reduction
    local_sum = warpReduceSum(local_sum);

    // Declare shared memory to hold one value per warp
    __shared__ float shared[32];  // 32 is sufficient for blocks up to 1024 threads (i.e., 1024/32 = 32 warps)

    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    // Each lane 0 writes its warp's sum to shared memory
    if (lane == 0) {
        shared[warpId] = local_sum;
    }
    __syncthreads();

    // Final reduction across warps in the block
    float block_sum = 0.0f;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < numWarps) {
        block_sum = shared[threadIdx.x];
    }
    if (threadIdx.x < warpSize) {
        block_sum = warpReduceSum(block_sum);
    }

    // Only one thread writes the block's result to global memory
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

// CUDA forward function
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Launch our combined kernel
    kl_div_kernel_combined<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Normalize by the total number of elements
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA combined optimized)");
}
