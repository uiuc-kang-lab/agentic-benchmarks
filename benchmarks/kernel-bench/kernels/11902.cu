#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    extern __shared__ float warp_sums[];
    
    float sum = 0.0f;
    const int start = (bid * blockDim.x + tid) * ELEMENTS_PER_THREAD;
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        const int idx = start + i;
        if (idx < n) {
            float log_pred = log_predictions[idx];
            sum += expf(log_pred) - targets[idx] * log_pred;
        }
    }

    sum = warp_reduce(sum);
    
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = lane < (BLOCK_SIZE/WARP_SIZE) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce(block_sum);
        
        if (lane == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int num_blocks = (n + (BLOCK_SIZE * ELEMENTS_PER_THREAD) - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD);
    const int shared_mem = (BLOCK_SIZE/WARP_SIZE) * sizeof(float);

    kl_div_kernel<<<num_blocks, BLOCK_SIZE, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA optimized reductions)");
}