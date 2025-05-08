#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int VECTOR_SIZE=4>
__global__ void kldiv_unrolled_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int64_t n) {

    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x * VECTOR_SIZE;
    float4 sum = {0};

    // Main vectorized loop with full unrolling
    #pragma unroll 2
    for (int64_t i = tid * VECTOR_SIZE; i < n - VECTOR_SIZE + 1; i += stride) {
        const float4 log_pred = *reinterpret_cast<const float4*>(&log_predictions[i]);
        const float4 target = *reinterpret_cast<const float4*>(&targets[i]);

        sum.x += __expf(log_pred.x) - target.x * log_pred.x;
        sum.y += __expf(log_pred.y) - target.y * log_pred.y;
        sum.z += __expf(log_pred.z) - target.z * log_pred.z;
        sum.w += __expf(log_pred.w) - target.w * log_pred.w;
    }

    // Handle remaining elements with unrolled scalar operations
    float remainder_sum = 0.0f;
    #pragma unroll
    for (int j = tid * VECTOR_SIZE + (n & ~3); j < n; j++) {
        const float lp = log_predictions[j];
        const float t = targets[j];
        remainder_sum += __expf(lp) - t * lp;
    }

    float thread_sum = sum.x + sum.y + sum.z + sum.w + remainder_sum;

    // Optimized warp reduction with full unrolling
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    __shared__ float block_sums[32];
    if (threadIdx.x % 32 == 0) {
        block_sums[threadIdx.x / 32] = thread_sum;
    }
    __syncthreads();

    // Final block reduction
    if (threadIdx.x == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < (blockDim.x + 31) / 32; i++) {
            total += block_sums[i];
        }
        atomicAdd(output, total);
    }
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {

    const int64_t n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    constexpr int THREADS = 512;
    const int blocks = (n + THREADS * 4 - 1) / (THREADS * 4);

    kldiv_unrolled_kernel<<<blocks, THREADS>>>(log_predictions.data_ptr<float>(),
                                            targets.data_ptr<float>(),
                                            output.data_ptr<float>(),
                                            n);

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL Divergence optimized forward");
}