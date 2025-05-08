#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Configuration constants
constexpr int THREADS_PER_BLOCK = 256;
constexpr int VECTOR_SIZE = 4;
constexpr int WARP_SIZE = 32;

__global__ void optimized_kldiv_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x * VECTOR_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int mask = 0xffffffff;
    
    // Use float4 for vectorized memory access
    float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Vectorized main computation loop
    #pragma unroll
    for (int i = tid * VECTOR_SIZE; i < n - VECTOR_SIZE + 1; i += stride) {
        float4 log_pred = *reinterpret_cast<const float4*>(&log_predictions[i]);
        float4 target = *reinterpret_cast<const float4*>(&targets[i]);
        
        sum.x += __expf(log_pred.x) - target.x * log_pred.x;
        sum.y += __expf(log_pred.y) - target.y * log_pred.y;
        sum.z += __expf(log_pred.z) - target.z * log_pred.z;
        sum.w += __expf(log_pred.w) - target.w * log_pred.w;
    }
    
    // Handle remaining elements
    int remainder = tid * VECTOR_SIZE + stride;
    while (remainder < n) {
        float log_pred = log_predictions[remainder];
        float target = targets[remainder];
        sum.x += __expf(log_pred) - target * log_pred;
        remainder++;
    }

    // Horizontal sum within thread
    float thread_sum = sum.x + sum.y + sum.z + sum.w;
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Block-level reduction using shared memory
    __shared__ float block_sum[WARP_SIZE];
    if (threadIdx.x % WARP_SIZE == 0) {
        block_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (threadIdx.x < WARP_SIZE) {
        float val = (threadIdx.x < blockDim.x / WARP_SIZE) ? block_sum[threadIdx.x] : 0.0f;
        
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }

        if (threadIdx.x == 0) {
            atomicAdd(output, val);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int blocks = (n + VECTOR_SIZE * THREADS_PER_BLOCK - 1) / (VECTOR_SIZE * THREADS_PER_BLOCK);

    optimized_kldiv_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Optimized KL divergence forward");
}