#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Warp-level reduction using __shfl_down_sync
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// The kernel computes the KL divergence and reduces the partial sums within each block.
__global__ void kl_div_kernel_shared_warp(const float* __restrict__ log_predictions,
                                           const float* __restrict__ targets,
                                           float* __restrict__ output,
                                           int n) {
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Grid-stride loop for balanced workload across threads
    for (int idx = global_idx; idx < n; idx += blockDim.x * gridDim.x) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
    }

    // Perform warp-level reduction using shuffle intrinsics
    sum = warp_reduce_sum(sum);

    // Allocate shared memory to store each warp's partial sum
    __shared__ float shared_warp[BLOCK_SIZE / WARP_SIZE];
    if ((tid % WARP_SIZE) == 0) {
        shared_warp[tid / WARP_SIZE] = sum;
    }
    __syncthreads();

    // Let the first warp perform the final reduction on the partial sums
    if (tid < (BLOCK_SIZE / WARP_SIZE)) {
        float warp_sum = shared_warp[tid];
        warp_sum = warp_reduce_sum(warp_sum);
        if (tid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

// CUDA forward function
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = BLOCK_SIZE;
    const int blocks = (n + threads - 1) / threads;

    // Launch the kernel
    kl_div_kernel_shared_warp<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA shared-warp optimized)");
}
