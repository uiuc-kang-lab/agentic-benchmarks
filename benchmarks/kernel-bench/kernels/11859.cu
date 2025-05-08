#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

__global__ void kldiv_optimized_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int64_t n) {

    // Calculate global thread index
    const int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    const int grid_stride = blockDim.x * gridDim.x * 4;

    float4 sum = {0, 0, 0, 0};

    // Vectorized memory access with float4
    for (int64_t i = idx; i < n; i += grid_stride) {
        float4 log_pred = *reinterpret_cast<const float4*>(&log_predictions[i]);
        float4 target = *reinterpret_cast<const float4*>(&targets[i]);

        sum.x += expf(log_pred.x) - target.x * log_pred.x;
        sum.y += expf(log_pred.y) - target.y * log_pred.y;
        sum.z += expf(log_pred.z) - target.z * log_pred.z;
        sum.w += expf(log_pred.w) - target.w * log_pred.w;
    }

    // Horizontal sum within thread
    float thread_sum = sum.x + sum.y + sum.z + sum.w;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

    // Block-level reduction
    __shared__ float block_sum[32];
    if (threadIdx.x % 32 == 0)
        block_sum[threadIdx.x / 32] = thread_sum;
    __syncthreads();

    // Final reduction and atomic add
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < blockDim.x / 32; i++)
            total += block_sum[i];
        atomicAdd(output, total);
    }
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {
    
    const int64_t n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Optimized launch configuration
    const int threads = 256;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);

    // Handle non-divisible sizes
    const int64_t padded_n = (n + 3) & ~3;
    kldiv_optimized_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        padded_n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence optimized forward");
}
