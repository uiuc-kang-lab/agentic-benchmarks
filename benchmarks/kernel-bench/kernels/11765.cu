#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int OPTIMAL_BLOCK_SIZE = 128;

__global__ void optimized_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_size = 4;
    const int vec_count = n / vec_size;

    float sum = 0.0f;

    // Vectorized processing with stride for coalescing
    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += stride) {
        const float4 log_vec = *reinterpret_cast<const float4*>(log_predictions + vec_idx*vec_size);
        const float4 tgt_vec = *reinterpret_cast<const float4*>(targets + vec_idx*vec_size);
        
        sum += expf(log_vec.x) - tgt_vec.x * log_vec.x;
        sum += expf(log_vec.y) - tgt_vec.y * log_vec.y;
        sum += expf(log_vec.z) - tgt_vec.z * log_vec.z;
        sum += expf(log_vec.w) - tgt_vec.w * log_vec.w;
    }

    // Process remaining elements
    const int scalar_base = vec_count * vec_size;
    for (int i = scalar_base + tid; i < n; i += stride) {
        sum += expf(log_predictions[i]) - targets[i] * log_predictions[i];
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Block-level reduction
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(output, sum);
    }
}

torch::Tensor optimized_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Optimized launch params for H100
    const int threads = OPTIMAL_BLOCK_SIZE;
    const int blocks = 512; 
    
    optimized_kl_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_kl_forward, "Block-optimized KL divergence (CUDA)");
}
