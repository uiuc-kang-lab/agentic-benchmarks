#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

__global__ void kl_div_kernel_unrolled(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const unsigned int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    // Compute local sum with manual loop unrolling
    float sum = 0.0f;
    
    // Process 4 elements per iteration
    const int unroll_factor = 4;
    const int unrolled_limit = n - (n % (stride * unroll_factor));
    
    // Unrolled loop
    for (int i = idx; i < unrolled_limit; i += stride * unroll_factor) {
        sum += compute_kl_element(log_predictions[i], targets[i]);
        sum += compute_kl_element(log_predictions[i + stride], targets[i + stride]);
        sum += compute_kl_element(log_predictions[i + stride * 2], targets[i + stride * 2]);
        sum += compute_kl_element(log_predictions[i + stride * 3], targets[i + stride * 3]);
    }
    
    // Handle remaining elements
    for (int i = idx + unrolled_limit; i < n; i += stride) {
        sum += compute_kl_element(log_predictions[i], targets[i]);
    }
    
    // Store in shared memory
    partial_sums[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory with unrolled loops
    #pragma unroll
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (unrolled)
    if (tid < 32) {
        volatile float* smem = partial_sums;
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {
            if (tid < offset) {
                smem[tid] += smem[tid + offset];
            }
        }
    }
    
    // Write result
    if (tid == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
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
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}