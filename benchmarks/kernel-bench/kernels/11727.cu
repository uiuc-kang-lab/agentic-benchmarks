#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE/WARP_SIZE)

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void optimized_kl_div_kernel(
    const float4* __restrict__ log_predictions4,
    const float4* __restrict__ targets4,
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n4,
    const int n) {
    
    // Calculate warp and lane IDs
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_count = gridDim.x * WARPS_PER_BLOCK;
    const int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    
    float sum = 0.0f;
    
    // Process aligned float4 elements (warp-aligned access)
    int base_idx = global_warp_id * WARP_SIZE + lane_id;
    #pragma unroll 4
    for (; base_idx < n4; base_idx += warp_count * WARP_SIZE) {
        float4 logp4 = __ldg(&log_predictions4[base_idx]);
        float4 targ4 = __ldg(&targets4[base_idx]);
        
        // Process all components without branching
        sum += expf(logp4.x) - targ4.x * logp4.x;
        sum += expf(logp4.y) - targ4.y * logp4.y;
        sum += expf(logp4.z) - targ4.z * logp4.z;
        sum += expf(logp4.w) - targ4.w * logp4.w;
    }
    
    // Process remaining elements (uniform across warp)
    const int remaining_start = n4 * 4;
    const int remaining_elements = n - remaining_start;
    base_idx = remaining_start + global_warp_id * WARP_SIZE + lane_id;
    
    if (remaining_elements > 0) {
        #pragma unroll 4
        for (; base_idx < n; base_idx += warp_count * WARP_SIZE) {
            float logp = __ldg(&log_predictions[base_idx]);
            float targ = __ldg(&targets[base_idx]);
            sum += expf(logp) - targ * logp;
        }
    }
    
    // Warp-level reduction (uniform operation)
    sum = warpReduceSum(sum);
    
    // Only first thread in each warp performs atomic add
    if (lane_id == 0) {
        atomicAdd(output, sum);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    const int n4 = n / 4;
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Ensure block size is multiple of warp size
    const int blocks = min(256, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Reinterpret data as float4 for vectorized access
    const float4* log_predictions4 = reinterpret_cast<const float4*>(log_predictions.data_ptr<float>());
    const float4* targets4 = reinterpret_cast<const float4*>(targets.data_ptr<float>());
    
    optimized_kl_div_kernel<<<blocks, BLOCK_SIZE>>>(
        log_predictions4,
        targets4,
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n4,
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}