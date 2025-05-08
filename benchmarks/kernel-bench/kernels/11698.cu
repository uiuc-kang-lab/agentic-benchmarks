#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel with coalesced memory access for KL divergence calculation
__global__ void coalesced_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;

    extern __shared__ float shared_warp_sums[];

    float sum = 0.0f;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Coalesced memory access: ensure threads in a warp access consecutive memory locations
    for (int i = global_tid; i < n; i += stride) {
        float log_pred = __ldg(&log_predictions[i]);
        float target = __ldg(&targets[i]);
        sum += __expf(log_pred) - target * log_pred;
    }

    // Intra-warp reduction using shuffle instructions
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Each warp's leader writes its result to shared memory
    if (lane_id == 0) {
        shared_warp_sums[warp_id] = sum;
    }

    // Synchronize to ensure all warp sums are written
    __syncthreads();

    // Let the first warp reduce the results from all warps in the block
    if (warp_id == 0) {
        float block_sum = (lane_id < warps_per_block) ? shared_warp_sums[lane_id] : 0.0f;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        // Thread 0 atomically accumulates the block's contribution to global output
        if (lane_id == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// Host function that sets up kernel launch parameters and invokes the kernel

torch::Tensor coalesced_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Dynamic block size selection based on the number of elements
    int best_block_size = 256; // Default
    if (n > 65536) {
        best_block_size = 512;
    } else if (n < 8192) {
        best_block_size = 128;
    }
    
    // Limit the number of blocks to avoid oversubscription
    const int max_blocks = 256;
    int blocks = min(max_blocks, (n + best_block_size - 1) / best_block_size);
    
    // Shared memory size: one float per warp
    int num_warps = best_block_size / 32;
    int shared_mem = num_warps * sizeof(float);

    coalesced_kl_div_kernel<<<blocks, best_block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &coalesced_kl_div_forward, "KLDivLoss with coalesced memory access (CUDA)");
}