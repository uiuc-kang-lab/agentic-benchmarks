#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce(float val) {
    // Warp-level reduction doesn't need __syncthreads()
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
    
    // Compute phase - no synchronization needed
    float thread_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x * 4 + tid;
    const int stride = blockDim.x * gridDim.x * 4;
    
    // Process 4 elements per iteration without sync
    while (idx + 3 < n) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int curr_idx = idx + i * blockDim.x;
            const float log_pred = log_predictions[curr_idx];
            const float target = targets[curr_idx];
            thread_sum += expf(log_pred) - target * log_pred;
        }
        idx += stride;
    }
    
    // Handle remaining elements
    while (idx < n) {
        const float log_pred = log_predictions[idx];
        const float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
        idx += blockDim.x;
    }
    
    // First reduce within warps - no sync needed
    thread_sum = warp_reduce(thread_sum);
    
    // Only the first thread in each warp writes to shared memory
    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    
    // Single sync point needed before cross-warp reduction
    __syncthreads();
    
    // Final reduction using first warp only
    if (warp_id == 0) {
        float warp_sum = 0.0f;
        if (lane_id < (blockDim.x >> 5)) { // number of warps
            warp_sum = sdata[lane_id];
        }
        
        // Reduce within final warp - no sync needed
        warp_sum = warp_reduce(warp_sum);
        
        // Single thread writes final result
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
    
    const int threads = 256;
    const int blocks = min((n + threads * 4 - 1) / (threads * 4), 1024);
    
    // Only need shared memory for warp results
    const int shared_mem = (threads >> 5) * sizeof(float);
    
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