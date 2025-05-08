#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void stride_optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    
    extern __shared__ float warp_sums[];
    
    float sum = 0.0f;
    
    // Calculate initial index and stride for this thread
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int items_per_thread = (n + total_threads - 1) / total_threads;
    
    // Each thread processes multiple elements with optimal stride pattern
    const int start_idx = tid;
    const int end_idx = min(start_idx + items_per_thread * total_threads, n);
    
    // Process elements with stride pattern for better cache utilization
    for (int i = start_idx; i < end_idx; i += total_threads) {
        float log_pred = __ldg(&log_predictions[i]);
        float target = __ldg(&targets[i]);
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0) {
        float warp_sum = (lane_id < warps_per_block) ? warp_sums[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor stride_optimized_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize block size based on input size
    int block_size = 256;
    if (n > 65536) block_size = 512;
    else if (n < 8192) block_size = 128;
    
    const int max_blocks = 256;
    const int blocks = min(max_blocks, (n + block_size - 1) / block_size);
    const int shared_mem = (block_size / 32) * sizeof(float);
    
    stride_optimized_kl_div_kernel<<<blocks, block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &stride_optimized_kl_div_forward, "Stride optimized KL divergence forward (CUDA)");
}