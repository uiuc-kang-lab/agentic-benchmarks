#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Modular device function to compute the KL divergence term for a given element
__device__ __forceinline__ float compute_kl_term(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Modular device function to accumulate a thread's local sum using a grid-stride loop
__device__ float compute_local_sum(const float* __restrict__ log_predictions,
                                   const float* __restrict__ targets,
                                   int n, int idx, int stride) {
    float sum = 0.0f;
    for (; idx < n; idx += stride) {
        sum += compute_kl_term(log_predictions[idx], targets[idx]);
    }
    return sum;
}

// Modular device function performing warp-level reduction using shuffle intrinsics
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Modular device function to perform block-level reduction using shared memory
__device__ float block_reduce_sum(float local_sum, volatile float* shared, int tid) {
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    if (lane == 0) {
        shared[wid] = local_sum;
    }
    __syncthreads();

    float block_sum = (tid < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (tid < blockDim.x / warpSize) {
        block_sum = warp_reduce_sum(block_sum);
    }
    return block_sum;
}

// Main kernel that utilizes modular device functions
__global__ void kl_div_kernel_modular(const float* __restrict__ log_predictions,
                                         const float* __restrict__ targets,
                                         float* __restrict__ output,
                                         const int n) {
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    // Each thread computes its local sum via a grid-stride loop
    float local_sum = compute_local_sum(log_predictions, targets, n, global_tid, stride);

    // Perform warp-level reduction
    local_sum = warp_reduce_sum(local_sum);

    // Shared memory allocation for block-level reduction
    extern __shared__ float shared[];
    float block_sum = block_reduce_sum(local_sum, shared, tid);

    // Thread 0 adds the block's contribution atomically
    if (tid == 0) {
        atomicAdd(output, block_sum);
    }
}

// PyTorch interface
torch::Tensor kl_div_cuda_forward(torch::Tensor log_predictions,
                                    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem_size = (threads / warpSize) * sizeof(float);

    kl_div_kernel_modular<<<blocks, threads, shared_mem_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA modular)");
}
