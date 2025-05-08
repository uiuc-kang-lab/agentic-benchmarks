#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void hierarchical_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Thread identification
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid % 32;
    const unsigned int warp_id = tid / 32;
    
    // Calculate global index with grid stride loop
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    const int stride = gridDim.x * BLOCK_SIZE;
    
    // Register for thread-local accumulation
    float local_sum = 0.0f;
    
    // Thread-level accumulation phase
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        local_sum += __expf(log_pred) - target * log_pred;
        idx += stride;
    }
    
    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // Shared memory for inter-warp reduction
    __shared__ float warp_sums[32]; // One element per warp
    
    // First thread in each warp writes its result
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        // Load warp sum or zero if out of bounds
        float sum = (tid < (BLOCK_SIZE / 32)) ? warp_sums[lane_id] : 0.0f;
        
        // Warp-level reduction of final results
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // First thread adds block result to global output
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    constexpr int BLOCK_SIZE = 256;
    const int num_blocks = min(65535, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    hierarchical_kl_div_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
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