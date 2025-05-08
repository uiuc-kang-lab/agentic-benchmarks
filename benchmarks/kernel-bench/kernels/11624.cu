#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute single KL divergence element using vectorized load4
__device__ __forceinline__ float4 compute_vectorized_kl4(
    const float4* __restrict__ log_predictions,
    const float4* __restrict__ targets,
    const int idx) {
    float4 log_pred = log_predictions[idx];
    float4 target = targets[idx];
    
    float4 result;
    result.x = expf(log_pred.x) - target.x * log_pred.x;
    result.y = expf(log_pred.y) - target.y * log_pred.y;
    result.z = expf(log_pred.z) - target.z * log_pred.z;
    result.w = expf(log_pred.w) - target.w * log_pred.w;
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

template<unsigned int BLOCK_SIZE=256>
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    extern __shared__ float sdata[];
    
    const unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + tid * 4;
    const unsigned int stride = blockDim.x * gridDim.x * 4;
    
    float thread_sum = 0.0f;
    
    // Process 4 elements at once using vectorized loads
    while (idx + 3 < n) {
        float4 result = compute_vectorized_kl4(
            reinterpret_cast<const float4*>(log_predictions),
            reinterpret_cast<const float4*>(targets),
            idx/4);
        thread_sum += result.x + result.y + result.z + result.w;
        idx += stride;
    }
    
    // Handle remaining elements
    while (idx < n) {
        thread_sum += expf(log_predictions[idx]) - targets[idx] * log_predictions[idx];
        idx++;
    }
    
    // Store in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Reduce within warps first
    if (tid < 32) {
        float warp_sum = 0.0f;
        #pragma unroll
        for (int i = tid; i < BLOCK_SIZE; i += 32) {
            warp_sum += sdata[i];
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
    
    kl_div_kernel<256><<<blocks, threads, shared_mem>>>(
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