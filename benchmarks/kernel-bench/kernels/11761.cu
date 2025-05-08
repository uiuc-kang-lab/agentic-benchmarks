#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constants in device constant memory
__constant__ int VECTOR_SIZE = 4;
__constant__ int WARP_SIZE = 32;

__global__ void full_warp_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int mask = 0xffffffff;
    
    float sum = 0.0f;
    
    // Vector loads for coalesced memory access
    const int vec_elements = n / 4;
    for (int i = tid; i < vec_elements; i += stride) {
        const float4 log_vec = reinterpret_cast<const float4*>(log_predictions)[i];
        const float4 tgt_vec = reinterpret_cast<const float4*>(targets)[i];
        
        // Process vector elements
        sum += expf(log_vec.x) - tgt_vec.x * log_vec.x;
        sum += expf(log_vec.y) - tgt_vec.y * log_vec.y;
        sum += expf(log_vec.z) - tgt_vec.z * log_vec.z;
        sum += expf(log_vec.w) - tgt_vec.w * log_vec.w;
    }
    
    // Handle remaining elements
    for (int i = tid + vec_elements * 4; i < n; i += stride) {
        sum += expf(log_predictions[i]) - targets[i] * log_predictions[i];
    }

    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Inter-warp reduction using ballot and shuffle
    if (lane_id == 0) {
        // Each warp's leader participates in the final reduction
        float warp_sum = sum;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(mask, warp_sum, offset);
            if (warp_id < offset) {
                warp_sum += other;
            }
        }
        
        // Only the first thread in the block does the atomic add
        if (warp_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor full_warp_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Optimize launch config for H100
    const int threads = 128; // 4 warps per block
    const int blocks = std::min(256, (n + threads - 1) / threads);
    
    full_warp_kl_div_kernel<<<blocks, threads, 0>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &full_warp_kl_div_forward, "Full warp KL divergence (CUDA)");
}