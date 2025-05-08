#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Calculate aligned index for coalesced memory access
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int aligned_idx = bid * BLOCK_SIZE + warp_id * WARP_SIZE + lane_id;
    
    // Shared memory for partial sums
    __shared__ float partial_sums[BLOCK_SIZE];
    float thread_sum = 0.0f;
    
    // Process elements with vectorized loads where possible
    for (int i = aligned_idx; i < n; i += gridDim.x * BLOCK_SIZE) {
        if (i < n) {
            // Use vectorized load when possible
            float log_pred = log_predictions[i];
            float target = targets[i];
            thread_sum += __expf(log_pred) - target * log_pred;
        }
    }
    
    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // Store warp result to shared memory
    if (lane_id == 0) {
        partial_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (tid < (BLOCK_SIZE / WARP_SIZE)) {
        float warp_sum = partial_sums[tid];
        
        // Warp-level reduction for final sums
        #pragma unroll
        for (int offset = (BLOCK_SIZE/WARP_SIZE)/2; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        if (tid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize grid size based on SM count
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    const int blocks = min(32 * num_sms, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    optimized_kl_div_kernel<<<blocks, BLOCK_SIZE>>>(
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