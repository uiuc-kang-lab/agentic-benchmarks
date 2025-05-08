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
    const float* log_predictions,
    const float* targets,
    float* output,
    int n) {
    
    // Grid-strided loop
    float thread_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction
    float warp_sum = warpReduceSum(thread_sum);
    
    // Store warp sums in shared memory
    extern __shared__ float warp_sums[];
    int warps_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // First warp reduces per-warp sums
    if (warp_id == 0) {
        float block_sum = (lane_id < warps_per_block) ? warp_sums[lane_id] : 0.0f;
        block_sum = warpReduceSum(block_sum);
        if (lane_id == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads_per_block = 256;
    int warps_per_block = threads_per_block / WARP_SIZE;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = warps_per_block * sizeof(float);
    
    kl_div_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL Div forward with minimal atomics");
}