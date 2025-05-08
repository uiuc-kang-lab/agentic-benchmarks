#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void vectorized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int warp_size = 32;
    const int vector_size = 4;
    float4 sum4 = {0, 0, 0, 0};

    // Vectorized index calculations
    const int vector_n = n / vector_size;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) * vector_size;
    const int stride = blockDim.x * gridDim.x * vector_size;

    // Process vectorized elements
    for (int i = tid; i < vector_n * vector_size; i += stride) {
        const float4 log_pred4 = *reinterpret_cast<const float4*>(log_predictions + i);
        const float4 target4 = *reinterpret_cast<const float4*>(targets + i);

        sum4.x += expf(log_pred4.x) - target4.x * log_pred4.x;
        sum4.y += expf(log_pred4.y) - target4.y * log_pred4.y;
        sum4.z += expf(log_pred4.z) - target4.z * log_pred4.z;
        sum4.w += expf(log_pred4.w) - target4.w * log_pred4.w;
    }

    // Process remaining elements
    float sum = sum4.x + sum4.y + sum4.z + sum4.w;
    for (int i = tid + vector_size * vector_n; i < n; i++) {
        sum += expf(log_predictions[i]) - targets[i] * log_predictions[i];
    }

    // Warp-level reduction
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Atomic add to global memory
    if (threadIdx.x % warp_size == 0) {
        atomicAdd(output, sum);
    }
}

torch::Tensor vectorized_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Optimized launch config for H100
    const int block_size = 256;
    const int max_blocks = 288;  // 144 SMs * 2
    const int blocks = min(max_blocks, (n + block_size * 4 - 1) / (block_size * 4));

    vectorized_kl_div_kernel<<<blocks, block_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &vectorized_kl_div_forward, "KLDivLoss with vectorized memory access (CUDA)");
}