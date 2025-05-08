#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute KL divergence contribution for a single element
__device__ __forceinline__ float compute_kl_div(const float log_pred, const float target) {
    // Use fast math exponential intrinsic for performance
    return __expf(log_pred) - target * log_pred;
}

// Device function to perform warp-level reduction using shuffle instructions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // Reduce within the warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Main kernel: modularized KL divergence kernel
__global__ void modularized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n
) {
    // Each thread computes a partial sum
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;

    // Grid-stride loop to accumulate contributions
    for (int i = global_tid; i < n; i += stride) {
        float log_val = __ldg(&log_predictions[i]);
        float target_val = __ldg(&targets[i]);
        sum += compute_kl_div(log_val, target_val);
    }

    // Perform warp-level reduction
    sum = warp_reduce_sum(sum);

    // Shared memory to hold the per-warp sums
    extern __shared__ float warp_sums[];
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // First warp loads the per-warp sums and reduces them
    float block_sum = 0.0f;
    int num_warps = blockDim.x / warpSize;
    if (threadIdx.x < num_warps) {
        block_sum = warp_sums[lane];
    }
    block_sum = warp_reduce_sum(block_sum);

    // First thread of the block adds the block contribution to the global sum
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

// Host function to launch the kernel
torch::Tensor modularized_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets
) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Dynamically choose block size based on problem size
    int block_size = 256;
    if (n > 65536) {
        block_size = 512;
    } else if (n < 8192) {
        block_size = 128;
    }

    const int max_blocks = 256;
    int blocks = min(max_blocks, (n + block_size - 1) / block_size);
    
    // Shared memory size: one float per warp
    int warps_per_block = block_size / 32;
    int shared_mem_size = warps_per_block * sizeof(float);

    modularized_kl_div_kernel<<<blocks, block_size, shared_mem_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &modularized_kl_div_forward, "Modularized KLDivLoss with modular device functions (CUDA)");
}
