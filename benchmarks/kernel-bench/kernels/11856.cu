#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define BLOCK_SIZE for experimentation; try 32, 64, 128, 256, or 512
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

// Kernel: compute KL divergence using vectorized loads, tunable block size, and shared memory reduction
__global__ void kldiv_blocksize_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int64_t n) {

    // Global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float thread_sum = 0.0f;

    // Process groups of 4 elements at a time
    int64_t n_vec = n / 4;  // number of complete float4 groups
    for (int64_t i = tid; i < n_vec; i += blockDim.x * gridDim.x) {
        // Use vectorized load for efficiency
        const float4* log_ptr = reinterpret_cast<const float4*>(log_predictions);
        const float4* target_ptr = reinterpret_cast<const float4*>(targets);
        float4 lp = log_ptr[i];
        float4 tgt = target_ptr[i];
        thread_sum += expf(lp.x) - tgt.x * lp.x;
        thread_sum += expf(lp.y) - tgt.y * lp.y;
        thread_sum += expf(lp.z) - tgt.z * lp.z;
        thread_sum += expf(lp.w) - tgt.w * lp.w;
    }

    // Process any remaining "tail" elements
    int64_t tail_start = n_vec * 4;
    for (int64_t i = tail_start + tid; i < n; i += blockDim.x * gridDim.x) {
        float lp = log_predictions[i];
        float t = targets[i];
        thread_sum += expf(lp) - t * lp;
    }

    // Warp-level reduction using shuffle
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Shared memory reduction within the block
    __shared__ float shared_data[BLOCK_SIZE];
    int lane = threadIdx.x;
    shared_data[lane] = thread_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            shared_data[lane] += shared_data[lane + stride];
        }
        __syncthreads();
    }

    // Atomic add from block-level reduction
    if (lane == 0) {
        atomicAdd(output, shared_data[0]);
    }
}

// CUDA forward function exposed to PyTorch
torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {

    const int64_t n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Determine grid dimensions based on the number of vectorized groups
    int64_t n_vec = n / 4;
    int blocks = (n_vec + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks < 32) { // Ensure a minimal number of blocks for good occupancy
        blocks = 32;
    }

    kldiv_blocksize_kernel<<<blocks, BLOCK_SIZE>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Block-size tuned KL divergence forward (CUDA)");
}
