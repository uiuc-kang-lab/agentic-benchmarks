#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
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
    const int n,
    const int elements_per_thread) {
    
    extern __shared__ float shared_data[];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int local_warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int warps_per_block = blockDim.x / 32;
    
    // Calculate starting index for this thread
    unsigned int i = (blockIdx.x * blockDim.x + tid) * elements_per_thread;
    const unsigned int grid_stride = blockDim.x * gridDim.x * elements_per_thread;
    
    float thread_sum = 0.0f;
    
    // Main computation loop with balanced workload
    while (i < n) {
        const int remaining = n - i;
        const int process_count = min(elements_per_thread, remaining);
        
        #pragma unroll
        for (int j = 0; j < process_count; j++) {
            const int idx = i + j;
            const float log_pred = log_predictions[idx];
            const float target = targets[idx];
            thread_sum += expf(log_pred) - target * log_pred;
        }
        
        i += grid_stride;
    }
    
    // First level reduction: warp-level using shuffle
    thread_sum = warp_reduce_sum(thread_sum);
    
    // Store warp results in shared memory
    if (lane_id == 0) {
        shared_data[local_warp_id] = thread_sum;
    }
    
    __syncthreads();
    
    // Second level reduction: across warps
    if (local_warp_id == 0) {
        float warp_sum = 0.0f;
        if (lane_id < warps_per_block) {
            warp_sum = shared_data[lane_id];
        }
        
        // Final warp-level reduction
        warp_sum = warp_reduce_sum(warp_sum);
        
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
    
    // Dynamic block size and elements per thread based on input size
    const int threads = 256;
    const int elements_per_thread = (n < 1024) ? 1 : 2;
    const int blocks = min((n + threads * elements_per_thread - 1) / (threads * elements_per_thread), 1024);
    
    // Shared memory size based on number of warps per block
    const int warps_per_block = threads / 32;
    const int shared_mem = warps_per_block * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        elements_per_thread
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}