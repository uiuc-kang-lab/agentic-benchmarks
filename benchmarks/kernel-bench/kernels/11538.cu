#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void optimized_kl_div_hybrid_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = tid / WARP_SIZE;  // Warp ID 
    const int lid = tid % WARP_SIZE;  // Lane ID
    const int gid = bid * BLOCK_SIZE + tid;
    
    // Use vectorized loads when possible
    using float4_t = float4;
    
    // Shared memory for partial sums
    __shared__ float partial_sums[BLOCK_SIZE];
    float thread_sum = 0.0f;

    // Vector loads for coalesced memory access
    #pragma unroll 4
    for (int i = gid; i < n-3; i += gridDim.x * BLOCK_SIZE) {
        if (i + 3 < n) {
            float4_t log_pred4 = reinterpret_cast<const float4_t*>(log_predictions)[i/4];
            float4_t target4 = reinterpret_cast<const float4_t*>(targets)[i/4];
            
            thread_sum += expf(log_pred4.x) - target4.x * log_pred4.x;
            thread_sum += expf(log_pred4.y) - target4.y * log_pred4.y;
            thread_sum += expf(log_pred4.z) - target4.z * log_pred4.z;
            thread_sum += expf(log_pred4.w) - target4.w * log_pred4.w;
        }
    }

    // Handle remaining elements
    for (int i = gid + (n/4)*4; i < n; i += gridDim.x * BLOCK_SIZE) {
        thread_sum += expf(log_predictions[i]) - targets[i] * log_predictions[i];
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Write warp results to shared memory
    if (lid == 0) {
        partial_sums[wid] = thread_sum;
    }
    __syncthreads();

    // Final reduction across warps
    if (tid < (BLOCK_SIZE/WARP_SIZE)) {
        float warp_sum = partial_sums[tid];
        
        // Block-level reduction for first warp
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
    
    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int shared_mem = BLOCK_SIZE * sizeof(float);
    
    optimized_kl_div_hybrid_kernel<<<blocks, BLOCK_SIZE, shared_mem>>>(
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