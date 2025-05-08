#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    
    extern __shared__ float smem[];
    
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    
    // Calculate initial index and strides
    int idx = blockIdx.x * blockDim.x + tid;
    const int grid_stride = gridDim.x * blockDim.x;
    
    // Each thread maintains its own sum
    float thread_sum = 0.0f;
    
    // Fast path for aligned access (processes 8 elements per iteration)
    const int aligned_n = n & ~7;
    
    #pragma unroll
    for (; idx < aligned_n; idx += grid_stride * 8) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int curr_idx = idx + i * grid_stride;
            if (curr_idx < n) {
                const float log_pred = log_predictions[curr_idx];
                const float target = targets[curr_idx];
                thread_sum += expf(log_pred) - target * log_pred;
            }
        }
    }
    
    // Handle remaining elements
    for (; idx < n; idx += grid_stride) {
        const float log_pred = log_predictions[idx];
        const float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction
    thread_sum = warp_reduce(thread_sum);
    
    // Store warp results to shared memory
    if (lane == 0) {
        smem[wid] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction using first warp
    if (wid == 0) {
        float warp_sum = 0.0f;
        if (lane < (blockDim.x >> 5)) { // number of warps
            warp_sum = smem[lane];
        }
        
        warp_sum = warp_reduce(warp_sum);
        
        if (lane == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize thread and block count based on input size
    const int threads = 256;
    const int min_elements_per_thread = 8;
    const int blocks = min((n + threads * min_elements_per_thread - 1) / 
                          (threads * min_elements_per_thread), 1024);
    
    // Shared memory for warp results
    const int shared_mem = (threads / 32) * sizeof(float);
    
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