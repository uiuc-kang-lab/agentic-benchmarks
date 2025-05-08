#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

__global__ void kldiv_coalesced_burst_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int64_t n) {

    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int vec_size = 4;
    const int64_t n_vec = (n + vec_size - 1) / vec_size;
    
    float4 sum = {0, 0, 0, 0};
    
    // Coalesced burst access pattern for aligned vectors
    for (int64_t vec_idx = tid; vec_idx < n_vec; vec_idx += gridDim.x * blockDim.x) {
        const int64_t offset = vec_idx * vec_size;
        
        if (offset + vec_size <= n) {
            const float4 log_pred = *reinterpret_cast<const float4*>(&log_predictions[offset]);
            const float4 target = *reinterpret_cast<const float4*>(&targets[offset]);
            
            sum.x += expf(log_pred.x) - target.x * log_pred.x;
            sum.y += expf(log_pred.y) - target.y * log_pred.y;
            sum.z += expf(log_pred.z) - target.z * log_pred.z;
            sum.w += expf(log_pred.w) - target.w * log_pred.w;
        } else {  // Handle partial vector
            const float* log_ptr = &log_predictions[offset];
            const float* tgt_ptr = &targets[offset];
            float tmp[vec_size] = {0};
            
            #pragma unroll
            for (int i = 0; i < vec_size; ++i) {
                if (offset + i < n) 
                    tmp[i] = expf(log_ptr[i]) - tgt_ptr[i] * log_ptr[i];
            }
            
            sum.x += tmp[0];
            sum.y += tmp[1];
            sum.z += tmp[2];
            sum.w += tmp[3];
        }
    }

    // Horizontal sum with instruction-level parallelism
    float thread_sum = sum.x + sum.y + sum.z + sum.w;

    // Warp-aware reduction without synchronization
    for (int offset = 16; offset >= 1; offset >>= 1)
        thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, offset);

    // Persistent thread-block reduction
    if (threadIdx.x % 32 == 0)
        atomicAdd(output, thread_sum);
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {
    
    const int64_t n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // H100-optimized launch config (4 warps per block)
    const int block_size = 128;
    const int grid_size = 144 * 8;  // 144 SMs * 8 active blocks per SM
    
    kldiv_coalesced_burst_kernel<<<grid_size, block_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence with coalesced memory access");
}
