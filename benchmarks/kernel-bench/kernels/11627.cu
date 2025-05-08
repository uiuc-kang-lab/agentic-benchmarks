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
    
    // Process 16 elements per thread for better instruction-level parallelism
    float thread_sum = 0.0f;
    
    // Use float4 for coalesced memory access
    float4* log_pred_vec = (float4*)log_predictions;
    float4* target_vec = (float4*)targets;
    
    const unsigned int elements_per_thread = 16;
    const unsigned int vec_elements = 4;
    unsigned int base_idx = (blockIdx.x * blockDim.x + tid) * elements_per_thread;
    const unsigned int stride = blockDim.x * gridDim.x * elements_per_thread;
    
    // Main loop using vectorized loads
    while (base_idx + elements_per_thread - 1 < n) {
        #pragma unroll
        for (int i = 0; i < elements_per_thread/vec_elements; i++) {
            const unsigned int vec_idx = (base_idx + i * vec_elements) / vec_elements;
            float4 log_pred4 = log_pred_vec[vec_idx];
            float4 target4 = target_vec[vec_idx];
            
            thread_sum += __expf(log_pred4.x) - target4.x * log_pred4.x;
            thread_sum += __expf(log_pred4.y) - target4.y * log_pred4.y;
            thread_sum += __expf(log_pred4.z) - target4.z * log_pred4.z;
            thread_sum += __expf(log_pred4.w) - target4.w * log_pred4.w;
        }
        base_idx += stride;
    }
    
    // Handle remaining elements
    unsigned int i = base_idx;
    while (i < n) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        thread_sum += __expf(log_pred) - target * log_pred;
        i++;
    }
    
    // Warp reduction
    thread_sum = warp_reduce(thread_sum);
    
    // Store warp results to shared memory
    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction using first warp
    if (warp_id == 0 && lane_id < (blockDim.x >> 5)) {
        float warp_sum = sdata[lane_id];
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
    
    // Use 128 threads per block for better occupancy
    const int threads = 128;
    const int min_blocks_per_sm = 8;
    const int max_blocks = 1024;
    const int blocks = min(
        (n + threads * 16 - 1) / (threads * 16),
        min(max_blocks, min_blocks_per_sm * 108) // H100 has 108 SMs
    );
    
    const int warps_per_block = threads / 32;
    const int shared_mem = warps_per_block * sizeof(float);
    
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