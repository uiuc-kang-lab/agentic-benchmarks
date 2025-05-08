#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int ELEMENTS_PER_THREAD = 8;

__global__ void blockwise_reduction_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    __shared__ float block_sum;
    if (threadIdx.x == 0) block_sum = 0.0f;
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Vectorized processing with stride pattern
    for (int i = tid * ELEMENTS_PER_THREAD; i < n; i += total_threads * ELEMENTS_PER_THREAD) {
        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            const int idx = i + j;
            if (idx < n) {
                const float log_pred = __ldg(log_predictions + idx);
                const float target = __ldg(targets + idx);
                thread_sum += expf(log_pred) - target * log_pred;
            }
        }
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

    // Block-level reduction
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(&block_sum, thread_sum);
    }
    __syncthreads();

    // Single atomicAdd per block
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

torch::Tensor blockwise_reduction_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 512;
    const int max_blocks = 128;
    const int blocks = min(max_blocks, (n + threads * ELEMENTS_PER_THREAD - 1) / (threads * ELEMENTS_PER_THREAD));

    blockwise_reduction_kl_kernel<<<blocks, threads, sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &blockwise_reduction_kl_forward, "Blockwise reduction KL divergence (CUDA)");
}