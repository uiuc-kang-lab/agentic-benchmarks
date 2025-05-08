#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)
#define ELEMENTS_PER_THREAD 4

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int grid_stride = gridDim.x * blockDim.x;
    
    // Each thread processes multiple elements
    float local_sum = 0.0f;
    
    // Grid-stride loop to handle large inputs
    for (int base_idx = tid * ELEMENTS_PER_THREAD; 
         base_idx < n; 
         base_idx += grid_stride * ELEMENTS_PER_THREAD) {
        
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int idx = base_idx + i;
            if (idx < n) {
                const float log_pred = __ldg(&log_predictions[idx]);
                const float target = __ldg(&targets[idx]);
                local_sum += expf(log_pred) - target * log_pred;
            }
        }
    }
    
    // Warp-level reduction
    float warp_sum = warpReduceSum(local_sum);
    
    // Only the first thread in each warp writes result
    if (lane_id == 0) {
        atomicAdd(output, warp_sum);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Calculate grid size based on input size and work per thread
    const int total_threads_needed = (n + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    const int blocks = min(
        (total_threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE,
        1024  // Conservative max blocks for better occupancy
    );
    
    kl_div_kernel<<<blocks, BLOCK_SIZE>>>(
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