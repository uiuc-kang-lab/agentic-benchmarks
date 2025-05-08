#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int ELEMENTS_PER_THREAD = 8;

__global__ void coalesced_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    float thread_sum = 0.0f;
    const int loop_stride = total_threads * ELEMENTS_PER_THREAD;
    
    for (int base = tid * ELEMENTS_PER_THREAD; base < n; base += loop_stride) {
        const int end = min(base + ELEMENTS_PER_THREAD, n);
        #pragma unroll
        for (int idx = base; idx < end; ++idx) {
            const float log_pred = __ldg(log_predictions + idx);
            const float target = __ldg(targets + idx);
            thread_sum += expf(log_pred) - target * log_pred;
        }
    }

    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    extern __shared__ float warp_sums[];
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < (blockDim.x/WARP_SIZE)) ? warp_sums[lane_id] : 0.0f;

        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor combined_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int max_blocks = (n + (threads * ELEMENTS_PER_THREAD) - 1) / (threads * ELEMENTS_PER_THREAD);
    const int blocks = std::min(512, max_blocks);
    const int shared_mem = (threads/WARP_SIZE) * sizeof(float);

    coalesced_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &combined_kl_forward, "Coalesced multi-element KL divergence with warp reduction (CUDA)");
}