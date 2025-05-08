#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute KL divergence for a single element
__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Optimized warp reduction using shuffle instructions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<unsigned int BLOCK_SIZE>
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    extern __shared__ float shared_mem[];
    
    const unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * BLOCK_SIZE * 4 + tid;
    const unsigned int gridSize = BLOCK_SIZE * 4 * gridDim.x;
    
    float thread_sum = 0.0f;
    
    // Vector loads for better memory coalescing
    float4 log_preds, targs;
    
    // Main computation loop with 4-element vectorized processing
    while (i + 3 * BLOCK_SIZE < n) {
        // Vector load log_predictions
        reinterpret_cast<float4*>(&log_preds)[0] = reinterpret_cast<const float4*>(&log_predictions[i])[0];
        // Vector load targets
        reinterpret_cast<float4*>(&targs)[0] = reinterpret_cast<const float4*>(&targets[i])[0];
        
        thread_sum += compute_kl_div(log_preds.x, targs.x);
        thread_sum += compute_kl_div(log_preds.y, targs.y);
        thread_sum += compute_kl_div(log_preds.z, targs.z);
        thread_sum += compute_kl_div(log_preds.w, targs.w);
        
        i += gridSize;
    }
    
    // Handle remaining elements
    while (i < n) {
        thread_sum += compute_kl_div(log_predictions[i], targets[i]);
        i += BLOCK_SIZE;
    }
    
    // Store in shared memory
    shared_mem[tid] = thread_sum;
    __syncthreads();
    
    // Two-level reduction: first at warp level using shuffle, then across warps
    if (tid < 32) {
        float warp_sum = 0.0f;
        if (tid < (BLOCK_SIZE / 32)) {
            #pragma unroll
            for (int i = tid * 32; i < (tid + 1) * 32; i++) {
                warp_sum += shared_mem[i];
            }
        }
        warp_sum = warp_reduce_sum(warp_sum);
        
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