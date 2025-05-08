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
    
    extern __shared__ float sdata[];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid & 31;
    const unsigned int warp_id = tid >> 5;
    
    // Calculate initial indices for different stride levels
    unsigned int idx_primary = blockIdx.x * blockDim.x + tid;
    const unsigned int grid_stride_primary = gridDim.x * blockDim.x;
    
    // Use float4 for vectorized memory access
    const float4* log_pred4 = reinterpret_cast<const float4*>(log_predictions);
    const float4* target4 = reinterpret_cast<const float4*>(targets);
    
    float thread_sum = 0.0f;
    
    // Primary stride loop - process 4 elements at a time
    const int vec_elements = n / 4;
    for (int i = idx_primary; i < vec_elements; i += grid_stride_primary) {
        float4 log_pred_vec = log_pred4[i];
        float4 target_vec = target4[i];
        
        // Process vector elements
        thread_sum += __expf(log_pred_vec.x) - target_vec.x * log_pred_vec.x;
        thread_sum += __expf(log_pred_vec.y) - target_vec.y * log_pred_vec.y;
        thread_sum += __expf(log_pred_vec.z) - target_vec.z * log_pred_vec.z;
        thread_sum += __expf(log_pred_vec.w) - target_vec.w * log_pred_vec.w;
    }
    
    // Secondary stride loop - handle remaining elements
    for (int i = vec_elements * 4 + tid; i < n; i += blockDim.x) {
        thread_sum += __expf(log_predictions[i]) - targets[i] * log_predictions[i];
    }
    
    // Warp-level reduction
    thread_sum = warp_reduce(thread_sum);
    
    // Store warp results
    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction using first warp
    if (warp_id == 0) {
        float warp_sum = 0.0f;
        if (lane_id < (blockDim.x >> 5)) { // number of warps
            warp_sum = sdata[lane_id];
        }
        
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
    
    // Optimize thread and block configuration
    const int threads = 256;
    const int vec_size = 4;
    const int min_elements_per_thread = 16;
    const int blocks = min((n + threads * min_elements_per_thread * vec_size - 1) / 
                          (threads * min_elements_per_thread * vec_size), 1024);
    
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