#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute single KL divergence element using vectorized operations
__device__ __forceinline__ float4 compute_vectorized_kl(
    const float4* __restrict__ log_pred4,
    const float4* __restrict__ target4) {
    float4 result;
    result.x = expf(log_pred4->x) - target4->x * log_pred4->x;
    result.y = expf(log_pred4->y) - target4->y * log_pred4->y;
    result.z = expf(log_pred4->z) - target4->z * log_pred4->z;
    result.w = expf(log_pred4->w) - target4->w * log_pred4->w;
    return result;
}

// Warp-level reduction using shuffle instructions
__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    extern __shared__ float shared_mem[];
    
    const unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + tid * 4;
    const unsigned int stride = blockDim.x * gridDim.x * 4;
    
    float thread_sum = 0.0f;
    
    // Process elements using float4 vectorized loads
    while (idx + 3 < n) {
        float4 log_pred4 = *reinterpret_cast<const float4*>(&log_predictions[idx]);
        float4 target4 = *reinterpret_cast<const float4*>(&targets[idx]);
        float4 result = compute_vectorized_kl(&log_pred4, &target4);
        thread_sum += result.x + result.y + result.z + result.w;
        idx += stride;
    }
    
    // Handle remaining elements
    while (idx < n) {
        thread_sum += expf(log_predictions[idx]) - targets[idx] * log_predictions[idx];
        idx++;
    }
    
    // Store in shared memory
    shared_mem[tid] = thread_sum;
    __syncthreads();
    
    // Two-level reduction: first within warps, then across warps
    if (tid < 32) {
        float warp_sum = 0.0f;
        if (tid < (blockDim.x / 32)) {
            #pragma unroll
            for (int i = tid * 32; i < (tid + 1) * 32 && i < blockDim.x; i++) {
                warp_sum += shared_mem[i];
            }
        }
        warp_sum = warp_reduce(warp_sum);
        
        if (tid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads * 4 - 1) / (threads * 4), 1024);
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}