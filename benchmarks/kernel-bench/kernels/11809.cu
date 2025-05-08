#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Warp-level reduction using shuffle
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel using grid-stride loop and combining shared memory reduction with warp-level primitives
__global__ void kl_div_kernel_optimized(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* output,
    const int n) {

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread accumulates its own sum using grid-stride loop
    float thread_sum = 0.0f;
    for (int idx = gid; idx < n; idx += stride) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
    }

    // Allocate shared memory for block-level reduction
    extern __shared__ float shared_sum[];
    shared_sum[tid] = thread_sum;
    __syncthreads();

    // Reduce using tree-based reduction until only 32 threads remain
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // Use warp-level reduction for the final 32 elements without further __syncthreads()
    if (tid < 32) {
        float sum = shared_sum[tid];
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Host function for launching the kernel
torch::Tensor kl_div_cuda_forward_optimized(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = std::min((n + threads - 1) / threads, 1024);
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
    m.def("forward", &kl_div_cuda_forward_optimized, "KL divergence forward (CUDA optimized with warp-level reduction)");
}
