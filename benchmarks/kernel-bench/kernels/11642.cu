#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for KL divergence calculation with unrolled loops
__global__ void kl_div_kernel_unrolled(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {
    
    // Get global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    // Unroll main computation loop - process 4 elements per iteration
    #pragma unroll 4
    for (int i = idx; i < n - 3; i += stride * 4) {
        if (i + stride * 3 < n) {
            // Process 4 elements
            float log_pred0 = log_predictions[i];
            float target0 = targets[i];
            float log_pred1 = log_predictions[i + stride];
            float target1 = targets[i + stride];
            float log_pred2 = log_predictions[i + stride * 2];
            float target2 = targets[i + stride * 2];
            float log_pred3 = log_predictions[i + stride * 3];
            float target3 = targets[i + stride * 3];
            
            sum += expf(log_pred0) - target0 * log_pred0;
            sum += expf(log_pred1) - target1 * log_pred1;
            sum += expf(log_pred2) - target2 * log_pred2;
            sum += expf(log_pred3) - target3 * log_pred3;
        }
    }
    
    // Handle remaining elements
    for (int i = idx + (n/4)*4; i < n; i += stride) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Store in shared memory
    partial_sums[threadIdx.x] = sum;
    __syncthreads();
    
    // Unrolled parallel reduction in shared memory
    #pragma unroll
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward_unrolled(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel_unrolled<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_unrolled, "KL divergence forward unrolled (CUDA)");
}