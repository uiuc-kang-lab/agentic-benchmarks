#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CUDA kernel with stride loop and warp-level reduction
__global__ void kl_div_kernel_stride(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n) {

    float thread_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple elements with a stride loop
    for (; idx < n; idx += stride) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
    }

    // Perform warp-level reduction
    thread_sum = warpReduceSum(thread_sum);

    // Allocate shared memory for warp sums
    extern __shared__ float warp_sums[];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        warp_sums[wid] = thread_sum;
    }
    __syncthreads();

    // Only threads in the first warp participate in further reduction
    float block_sum = 0.0f;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (threadIdx.x < num_warps) {
        block_sum = warp_sums[threadIdx.x];
    } else {
        block_sum = 0.0f;
    }
    block_sum = warpReduceSum(block_sum);

    // Thread 0 writes the block's contribution to the final output
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

// PyTorch interface
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();

    // Allocate output tensor
    auto output = torch::zeros({1}, log_predictions.options());

    // Launch parameters
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    int warpsPerBlock = (threads + WARP_SIZE - 1) / WARP_SIZE;
    int shared_mem = warpsPerBlock * sizeof(float);

    // Launch CUDA kernel
    kl_div_kernel_stride<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA optimized using stride loops)");
}
