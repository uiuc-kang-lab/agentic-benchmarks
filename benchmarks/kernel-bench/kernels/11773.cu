#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define warp size constant
constexpr int WARP_SIZE = 32;

// Kernel: Optimizes global memory loads using __ldg() and vectorized float4 loads
__global__ void aligned_ldg_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;

    // Process 128-bit aligned loads using float4
    int num_vectorized = n / 4;  // Number of complete groups of 4 floats
    for (int i = tid; i < num_vectorized; i += total_threads) {
        // Load 4 floats at a time using __ldg to leverage read-only cache
        float4 lp = __ldg(reinterpret_cast<const float4*>(log_predictions) + i);
        float4 t  = __ldg(reinterpret_cast<const float4*>(targets) + i);

        sum += expf(lp.x) - t.x * lp.x;
        sum += expf(lp.y) - t.y * lp.y;
        sum += expf(lp.z) - t.z * lp.z;
        sum += expf(lp.w) - t.w * lp.w;
    }

    // Handle any remaining elements
    int remaining_start = num_vectorized * 4;
    for (int i = tid; i < (n - remaining_start); i += total_threads) {
        int idx = remaining_start + i;
        float lp = __ldg(log_predictions + idx);
        float t  = __ldg(targets + idx);
        sum += expf(lp) - t * lp;
    }

    // Intra-warp reduction using shuffle operations
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Shared memory for block-level reduction: one float per warp
    extern __shared__ float smem[];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction by the first warp in the block
    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        sum = smem[threadIdx.x];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Host function
torch::Tensor aligned_ldg_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    // Calculate number of blocks based on vectorized processing
    int blocks = (n / 4 + threads - 1) / threads;
    blocks = min(blocks, 256); // Limit blocks to ensure enough work per block
    int shared_mem = (threads / WARP_SIZE) * sizeof(float);

    aligned_ldg_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &aligned_ldg_kl_forward, "KL divergence optimized with __ldg and 128-bit aligned accesses (CUDA)");
}
