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
    
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid & 31;
    const unsigned int warp_id = tid >> 5;
    
    // Calculate initial indices for this thread
    unsigned int base_idx = blockIdx.x * blockDim.x * 16 + tid;
    const unsigned int grid_stride = gridDim.x * blockDim.x * 16;
    
    float thread_sum = 0.0f;
    
    // Fast path - process 16 elements per thread per iteration
    const int aligned_n = (n / 16) * 16;
    
    #pragma unroll 4
    for (; base_idx < aligned_n; base_idx += grid_stride) {
        #pragma unroll
        for (int i = 0; i < 16; i += 4) {
            const int curr_idx = base_idx + i;
            
            // Load 4 elements at once
            float4 log_pred4 = *reinterpret_cast<const float4*>(&log_predictions[curr_idx]);
            float4 target4 = *reinterpret_cast<const float4*>(&targets[curr_idx]);
            
            // Process elements
            thread_sum += __expf(log_pred4.x) - target4.x * log_pred4.x;
            thread_sum += __expf(log_pred4.y) - target4.y * log_pred4.y;
            thread_sum += __expf(log_pred4.z) - target4.z * log_pred4.z;
            thread_sum += __expf(log_pred4.w) - target4.w * log_pred4.w;
        }
    }
    
    // Handle remaining elements with boundary checking
    base_idx = (aligned_n / 16) * 16 + tid;
    while (base_idx < n) {
        const float log_pred = log_predictions[base_idx];
        const float target = targets[base_idx];
        thread_sum += __expf(log_pred) - target * log_pred;
        base_idx += blockDim.x;
    }
    
    // Warp-level reduction
    thread_sum = warp_reduce(thread_sum);
    
    // Store warp results
    if (lane_id == 0) {
        smem[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction using first warp
    if (warp_id == 0 && lane_id < (blockDim.x >> 5)) {
        float warp_sum = smem[lane_id];
        warp_sum = warp_reduce(warp_sum);
        
        if (lane_id == 0) {
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
    const int min_elements_per_thread = 16;
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