#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kl_div_kernel_sync_optimized(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n) {

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    extern __shared__ float shared_sum[];

    float local_sum = 0.0f;
    for (int idx = gid; idx < n; idx += stride) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        local_sum += __expf(log_pred) - target * log_pred;
    }

    shared_sum[tid] = local_sum;
    __syncthreads(); // Synchronize only when threads have updated shared memory

    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        if (s > 1) __syncthreads(); // Only synchronize when there are more reductions to perform
    }

    if (tid == 0) {
        atomicAdd(output, shared_sum[0]);
    }
}

torch::Tensor kl_div_cuda_forward_sync_optimized(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);

    kl_div_kernel_sync_optimized<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_sync_optimized, "Optimized KL divergence forward (CUDA)");
}