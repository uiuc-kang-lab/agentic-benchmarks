#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 128;
constexpr int WARP_SIZE = 32;

__global__ void aligned_kldiv_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int64_t n) {
    
    // Ensure 128-bit alignment for memory accesses
    static_assert(sizeof(float4) == 16, "float4 must be 16 bytes");
    
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t gid = bid * BLOCK_SIZE + tid;
    const uint32_t stride = gridDim.x * BLOCK_SIZE;
    
    float local_sum = 0.0f;
    
    // Process 4 elements per thread using aligned loads
    #pragma unroll 4
    for (uint32_t idx = gid * 4; idx < n; idx += stride * 4) {
        // Use __ldg for read-only memory access
        if (idx + 3 < n) {
            float4 log_pred;
            float4 target;
            
            // Aligned loads using __ldg
            log_pred.x = __ldg(log_predictions + idx);
            log_pred.y = __ldg(log_predictions + idx + 1);
            log_pred.z = __ldg(log_predictions + idx + 2);
            log_pred.w = __ldg(log_predictions + idx + 3);
            
            target.x = __ldg(targets + idx);
            target.y = __ldg(targets + idx + 1);
            target.z = __ldg(targets + idx + 2);
            target.w = __ldg(targets + idx + 3);
            
            // Compute KL divergence for 4 elements
            local_sum += __expf(log_pred.x) - target.x * log_pred.x;
            local_sum += __expf(log_pred.y) - target.y * log_pred.y;
            local_sum += __expf(log_pred.z) - target.z * log_pred.z;
            local_sum += __expf(log_pred.w) - target.w * log_pred.w;
        } else {
            // Handle remaining elements
            for (uint32_t i = 0; i < 4 && idx + i < n; ++i) {
                float log_pred = __ldg(log_predictions + idx + i);
                float target = __ldg(targets + idx + i);
                local_sum += __expf(log_pred) - target * log_pred;
            }
        }
    }
    
    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // Shared memory for warp results
    __shared__ float warp_sums[4];  // Supports up to 128 threads (4 warps)
    
    const uint32_t lane = tid % WARP_SIZE;
    const uint32_t wid = tid / WARP_SIZE;
    
    if (lane == 0) {
        warp_sums[wid] = local_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (tid < 4) {
        float warp_sum = warp_sums[tid];
        
        #pragma unroll
        for (int offset = 2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        if (tid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {
    
    const auto n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int num_blocks = (n + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);
    
    aligned_kldiv_kernel<<<num_blocks, BLOCK_SIZE>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Aligned KL divergence forward");
}