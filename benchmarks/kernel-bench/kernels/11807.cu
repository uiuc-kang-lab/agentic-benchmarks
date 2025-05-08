#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using warp shuffle reduction to minimize synchronization overhead
__global__ void kl_div_kernel_warp_reduce(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread computes its partial sum using grid-stride loop
    float thread_sum = 0.0f;
    for (int idx = gid; idx < n; idx += stride) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
    }

    // Use warp shuffle reduction to sum values within a warp
    unsigned int mask = 0xffffffff; // full warp mask
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Now, the first thread (lane 0) of each warp holds the sum of that warp
    int lane = tid & (warpSize - 1); // tid % warpSize
    int warp_id = tid / warpSize;

    // Allocate shared memory for warp-level partial sums
    extern __shared__ float shared_warp[];
    if (lane == 0) {
        shared_warp[warp_id] = thread_sum;
    }
    __syncthreads();

    // Let the first warp finalize the block-level reduction
    float block_sum = 0.0f;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (tid < num_warps) {
        block_sum = shared_warp[tid];
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (tid == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// The forward function calling the optimized kernel

torch::Tensor kl_div_cuda_forward_warp_reduce(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = std::min((n + threads - 1) / threads, 1024);

    // Shared memory size: one float per warp
    int num_warps = (threads + warpSize - 1) / warpSize; 
    const int shared_mem = num_warps * sizeof(float);

    kl_div_kernel_warp_reduce<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Normalize the output by dividing with n
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_warp_reduce, "KL divergence forward with warp shuffle reduction (CUDA)");
}
