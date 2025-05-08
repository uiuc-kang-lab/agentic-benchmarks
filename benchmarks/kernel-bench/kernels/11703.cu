#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Optimized CUDA kernel for KL divergence using warp-level primitives,
// shared memory reduction and dynamic block size selection.
__global__ void optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;

    float sum = 0.0f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Loop over data using grid-stride loop to accumulate local sum
    for (int i = tid; i < n; i += stride) {
        float log_pred = __ldg(&log_predictions[i]);
        float target = __ldg(&targets[i]);
        sum += expf(log_pred) - target * log_pred;
    }

    // Intra-warp reduction using shfl_down_sync
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Allocate shared memory for warp results
    extern __shared__ float warp_results[];

    // Write each warp's sum into shared memory (only lane 0 in each warp)
    if (lane_id == 0) {
        warp_results[warp_id] = sum;
    }
    __syncthreads();

    // Let the first warp perform another reduction over warp results
    if (warp_id == 0) {
        float warp_sum = (threadIdx.x < warps_per_block) ? warp_results[threadIdx.x] : 0.0f;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        // Only one thread writes the block's partial result with atomicAdd
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

// Host function with dynamic block size selection
torch::Tensor optimized_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Dynamically choose block size based on problem size
    int best_block_size = 256;
    const int max_blocks = 256;
    if (n > 65536) {
        best_block_size = 512;
    } else if (n < 8192) {
        best_block_size = 128;
    }
    
    int num_warps = best_block_size / 32;
    int blocks = std::min(max_blocks, (n + best_block_size - 1) / best_block_size);
    int shared_mem = num_warps * sizeof(float);

    optimized_kl_div_kernel<<<blocks, best_block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Normalize the accumulated sum
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_kl_div_forward, "Optimized KL divergence forward (CUDA)");
}
