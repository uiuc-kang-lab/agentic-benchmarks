#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__
float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int grid_stride = gridDim.x * blockDim.x;
    
    // Shared memory for partial sums (one per warp)
    extern __shared__ float warp_sums[];
    
    // Calculate starting index for this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // Process elements with grid stride loop
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
        idx += grid_stride;
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    
    // First thread in each warp writes the result
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction across warps (done by first warp)
    if (warp_id == 0) {
        sum = (lane_id < warps_per_block) ? warp_sums[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Calculate optimal thread/block configuration
    const int threads_per_block = 256; // Increased to improve parallelism
    const int max_blocks = 256; // Increased to allow more blocks
    const int num_blocks = min(max_blocks, (n + threads_per_block - 1) / threads_per_block);
    
    // Shared memory size (one float per warp)
    const int warps_per_block = threads_per_block / 32;
    const int shared_mem = warps_per_block * sizeof(float);
    
    auto output = torch::zeros({1}, log_predictions.options());
    
    optimized_kl_div_kernel<<<num_blocks, threads_per_block, shared_mem>>>(
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