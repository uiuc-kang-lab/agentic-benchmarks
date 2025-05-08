#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle intrinsic with unrolled loop
__inline__ __device__ float warpReduceSum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// Optimized kernel for KL divergence using __ldg for read-only accesses and vectorized loads
__global__ void kl_div_kernel_ldg(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* output,
    const int n) {

    float local_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Process in vectorized fashion using float4 (128-bit aligned loads)
    int vec_n = n / 4;  // Number of complete float4 elements
    const float4* log_preds_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targets_vec   = reinterpret_cast<const float4*>(targets);

    for (int i = idx; i < vec_n; i += stride) {
        // Use __ldg to load from global read-only memory fast
        float4 lp4 = __ldg(log_preds_vec + i);
        float4 tg4 = __ldg(targets_vec + i);
        local_sum += expf(lp4.x) - tg4.x * lp4.x;
        local_sum += expf(lp4.y) - tg4.y * lp4.y;
        local_sum += expf(lp4.z) - tg4.z * lp4.z;
        local_sum += expf(lp4.w) - tg4.w * lp4.w;
    }

    // Process remaining elements
    int remaining_start = vec_n * 4;
    for (int i = remaining_start + idx; i < n; i += stride) {
        float lp = __ldg(log_predictions + i);
        float tg = __ldg(targets + i);
        local_sum += expf(lp) - tg * lp;
    }

    // Warp-level reduction
    local_sum = warpReduceSum(local_sum);

    // Shared memory reduction among warps in the block
    __shared__ float shared[32];  // Enough for up to 1024 threads (32 warps)
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[warpId] = local_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < numWarps) {
        block_sum = shared[threadIdx.x];
    }
    if (threadIdx.x < warpSize) {
        block_sum = warpReduceSum(block_sum);
    }

    // Write the block's result to global memory using atomic add
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

    kl_div_kernel_ldg<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward using __ldg and aligned loads (CUDA)");
}
