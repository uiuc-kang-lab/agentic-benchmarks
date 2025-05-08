#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 512

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_optimized(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    extern __shared__ float shared_partial_sums[];
    const int tid = threadIdx.x;
    const int global_tid = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Stride loop to ensure all elements are processed by the blocks
    for (int i = global_tid; i < n; i += blockDim.x * gridDim.x) {
        const float log_pred = log_predictions[i];
        const float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }

    // First level reduction across the block
    shared_partial_sums[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_partial_sums[tid] += shared_partial_sums[tid + s];
        }
        __syncthreads();
    }

    // Atomic add to global memory, if this is the first thread in the block
    if (tid == 0) {
        atomicAdd(output, shared_partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward_optimized(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    int threads = std::min(MAX_BLOCK_SIZE, n);
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);

    kl_div_kernel_optimized<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_optimized", &kl_div_cuda_forward_optimized, "KL divergence forward (CUDA optimized kernel)");
}