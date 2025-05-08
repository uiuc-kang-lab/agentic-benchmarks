#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that computes KL divergence with loop unrolling using #pragma unroll
__global__ void kldiv_unroll_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int64_t n) {

    // Calculate global thread index and total threads
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Process data in groups of 4 using vectorized loads
    // Compute the largest multiple of 4 that is <= n
    int64_t n4 = n - (n & 3);
    float local_sum = 0.0f;
    int stride = total_threads * 4;

    // Vectorized loop: each iteration processes 4 elements
    for (int64_t i = tid * 4; i < n4; i += stride) {
        // Load 4 elements at once
        float4 lp = *reinterpret_cast<const float4*>(&log_predictions[i]);
        float4 tgt = *reinterpret_cast<const float4*>(&targets[i]);
        
        // Manually unroll the computation for the four elements
        #pragma unroll
        {
            local_sum += expf(lp.x) - tgt.x * lp.x;
            local_sum += expf(lp.y) - tgt.y * lp.y;
            local_sum += expf(lp.z) - tgt.z * lp.z;
            local_sum += expf(lp.w) - tgt.w * lp.w;
        }
    }

    // Process remaining tail elements
    for (int64_t i = n4 + tid; i < n; i += total_threads) {
        float lp = log_predictions[i];
        float tgt = targets[i];
        local_sum += expf(lp) - tgt * lp;
    }

    // Warp-level reduction using shuffle, unrolled for fixed warpSize (32)
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Each warp's leader writes its result in shared memory
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    if (lane == 0) {
        smem[warpId] = local_sum;
    }
    __syncthreads();

    // Final block-level reduction performed by thread 0
    if (threadIdx.x == 0) {
        int numWarps = (blockDim.x + 31) >> 5;
        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < numWarps; i++) {
            block_sum += smem[i];
        }
        atomicAdd(output, block_sum);
    }
}

// CUDA function exposed to PyTorch
torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {

    const int64_t n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    kldiv_unroll_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence unrolled forward (CUDA)");
}
