#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int n) {
    
    // Aligned 128-bit accesses through __ldg and grid-striding
    float thread_sum = 0.0f;
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        // Use read-only cache intrinsic with aligned access
        const float log_pred = __ldg(log_predictions + i);
        const float target = __ldg(targets + i);
        thread_sum += expf(log_pred) - target * log_pred;
    }

    // Warp-level reduction
    float warp_sum = warpReduceSum(thread_sum);

    // Shared memory for block reduction
    extern __shared__ float warp_sums[];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    // First warp reduces block sum
    if (warp_id == 0) {
        float block_sum = (lane_id < (blockDim.x/WARP_SIZE)) ? warp_sums[lane_id] : 0.0f;
        block_sum = warpReduceSum(block_sum);
        
        if (threadIdx.x == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // 256 threads per block, aligned to 128-bit boundaries
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const size_t shared_mem = (threads/WARP_SIZE) * sizeof(float);

    kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL Div forward with LDG optimizations");
}