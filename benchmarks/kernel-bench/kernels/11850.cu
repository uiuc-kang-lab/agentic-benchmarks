#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel using cooperative groups and improved memory access
__global__ void optimized_kldiv_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Use cooperative groups for better synchronization
    using namespace cooperative_groups;
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);

    const unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Coalesced memory access with longer stride
    #pragma unroll 4
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        // Fused computation for better instruction throughput
        sum += __expf(log_pred) - target * log_pred;
        idx += gridDim.x * blockDim.x;
    }

    // Warp-level reduction using cooperative groups
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        sum += warp.shfl_down(sum, offset);
    }

    // Shared memory for block-level reduction
    __shared__ float warp_results[32];

    // Only first thread in each warp writes to shared memory
    if (warp.thread_rank() == 0) {
        warp_results[tid / 32] = sum;
    }

    block.sync();

    // Final reduction by first warp
    if (tid < 32) {
        float block_sum = (tid < (block.group_dim().x + 31) / 32) ? warp_results[tid] : 0.0f;
        
        // Warp-level reduction for final result
        for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
            block_sum += warp.shfl_down(block_sum, offset);
        }

        // Single atomic add per block
        if (warp.thread_rank() == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Optimal thread/block configuration
    const int threads = 512;  // Maximum threads for better occupancy
    const int blocks = min(65535, (n + threads - 1) / threads);

    optimized_kldiv_kernel<<<blocks, threads>>>(
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