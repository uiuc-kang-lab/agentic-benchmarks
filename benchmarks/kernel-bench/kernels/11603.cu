#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce(float val) {
    // Warp-level reduction without synchronization
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
    
    extern __shared__ float shared[];
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    
    // Compute phase - accumulate local sum
    float thread_sum = 0.0f;
    
    // Process elements with grid stride
    for (int idx = blockIdx.x * blockDim.x + tid; idx < n; idx += blockDim.x * gridDim.x) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
    }
    
    // First level reduction - warp reduction (no sync needed)
    thread_sum = warp_reduce(thread_sum);
    
    // Write warp results to shared memory
    if (lane_id == 0) {
        shared[warp_id] = thread_sum;
    }
    
    // Single sync point needed before reading from shared memory
    __syncthreads();
    
    // Final reduction - only first warp
    if (tid < 32) {
        float sum = (tid < (blockDim.x >> 5)) ? shared[tid] : 0.0f;
        sum = warp_reduce(sum);
        
        if (tid == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize thread count to be multiple of warp size
    const int threads = 256;  // 8 warps per block
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = (threads / 32) * sizeof(float); // Space for warp results
    
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