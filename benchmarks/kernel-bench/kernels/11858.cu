#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

template<int VEC_SIZE>
__global__ void kldiv_warp_aligned_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int64_t n) {

    const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warps_per_block = blockDim.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;

    float sum = 0.0f;
    
    // Process 4 elements per thread using aligned vector accesses
    const int64_t vec_n = n / VEC_SIZE;
    for (int64_t i = gtid; i < vec_n; i += gridDim.x * blockDim.x) {
        float4 pred = *reinterpret_cast<const float4*>(&log_predictions[i*VEC_SIZE]);
        float4 tgt = *reinterpret_cast<const float4*>(&targets[i*VEC_SIZE]);
        
        sum += expf(pred.x) - tgt.x * pred.x;
        sum += expf(pred.y) - tgt.y * pred.y;
        sum += expf(pred.z) - tgt.z * pred.z;
        sum += expf(pred.w) - tgt.w * pred.w;
    }

    // Handle remaining elements without branches
    const int64_t scalar_base = vec_n * VEC_SIZE;
    const int64_t scalar_idx = scalar_base + gtid;
    if (scalar_idx < n) {
        float pred = log_predictions[scalar_idx];
        float tgt = targets[scalar_idx];
        sum += expf(pred) - tgt * pred;
    }

    // Warp-aware reduction without shared memory
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // First thread in warp aggregates and atomically adds
    if (lane_id == 0)
        atomicAdd(output, sum);
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {
    
    const int64_t n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    constexpr int VEC_SIZE = 4;
    const uint32_t threads = 256;  // 8 warps per block
    const uint32_t blocks = 144 * 6; // 144 SMs * 6 resident blocks

    kldiv_warp_aligned_kernel<VEC_SIZE><<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence warp-aligned forward");
}