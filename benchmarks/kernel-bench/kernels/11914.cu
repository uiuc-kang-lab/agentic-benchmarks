#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// Warp-level reduction using shuffle operations with uniform control flow
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel that minimizes warp divergence by refactoring the loop to have a uniform workload
// Each thread computes a base number of iterations (which are guaranteed in-bound) without any conditionals,
// then a single conditional (outside the loop) handles the remaining elements. This keeps most of the control flow uniform.
__global__ void kl_div_kernel_uniform(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Compute the number of iterations that every thread can perform without branching
    int base_iters = n / total_threads;  // Guaranteed full iterations
    int remainder   = n - base_iters * total_threads;  // Extra elements

    float thread_sum = 0.0f;

    // Process the bulk of elements in a uniform loop without conditionals
    for (int i = 0; i < base_iters; i++) {
        int idx = i * total_threads + global_tid;
        float log_val = log_predictions[idx];
        float target = targets[idx];
        thread_sum += expf(log_val) - target * log_val;
    }

    // Handle the remainder in a single conditional to avoid divergence in the main loop
    if (global_tid < remainder) {
        int idx = base_iters * total_threads + global_tid;
        float log_val = log_predictions[idx];
        float target = targets[idx];
        thread_sum += expf(log_val) - target * log_val;
    }

    // Intra-warp reduction
    float sum = warp_reduce_sum(thread_sum);

    // Use shared memory to combine results from each warp in the block
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction by the first warp in the block
    if (threadIdx.x < (BLOCK_SIZE / WARP_SIZE)) {
        float block_sum = warp_sums[threadIdx.x];
        block_sum = warp_reduce_sum(block_sum);
        if (threadIdx.x == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// Host function that sets up and launches the CUDA kernel
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Choose grid dimensions so that total threads are close to n to minimize extra conditional load overhead
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Note: total threads used in kernel
    int total_threads = blocks * BLOCK_SIZE;

    kl_div_kernel_uniform<<<blocks, BLOCK_SIZE>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA uniform control flow)");
}
