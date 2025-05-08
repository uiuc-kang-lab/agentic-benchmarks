#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

// Warp-level reduction using shuffle down intrinsic
__device__ __forceinline__ float warp_reduce(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fused kernel: combines grid-stride looping with loop unrolling and warp-level reduction
__global__ void kl_div_kernel_fused(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    int grid_stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    // Each thread processes ELEMENTS_PER_THREAD contiguous elements per iteration
    // and uses a grid-stride loop to cover the full data range
    for (int start = global_tid * ELEMENTS_PER_THREAD; start < n; start += grid_stride * ELEMENTS_PER_THREAD) {
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int idx = start + i;
            if (idx < n) {
                float log_pred = log_predictions[idx];
                float target = targets[idx];
                local_sum += expf(log_pred) - target * log_pred;
            }
        }
    }

    // Intra-warp reduction via shuffle
    local_sum = warp_reduce(local_sum);

    // Each warp's leading thread writes its result to shared memory
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction: first warp combines results from all warps of the block
    if (tid < (blockDim.x / WARP_SIZE)) {
        float sum = warp_sums[tid];
        sum = warp_reduce(sum);
        if (tid == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Host function called from PyTorch
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Determine grid dimensions based on number of elements and ELEMENTS_PER_THREAD per thread
    const int threads = BLOCK_SIZE;
    const int total_threads = (n + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    const int blocks = (total_threads + threads - 1) / threads;

    kl_div_kernel_fused<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA fused)");
}
