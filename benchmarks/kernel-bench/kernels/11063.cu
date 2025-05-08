#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void cross_entropy_loss_kernel_optimized(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    const int batch_size,
    const int num_classes
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int sample_idx = bid;
    
    if (sample_idx >= batch_size) return;
    
    const float* logits_sample = logits + sample_idx * num_classes;
    const int target = targets[sample_idx];
    
    // Use multiple threads per sample for parallel reduction
    float thread_max = -INFINITY;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        thread_max = fmaxf(thread_max, logits_sample[i]);
    }
    
    // Warp-level reduction for maximum
    float max_logit = warp_reduce_max(thread_max);
    if (tid == 0) {
        __shared__ float s_max;
        s_max = max_logit;
        max_logit = s_max; // Broadcast to all threads
    }
    __syncthreads();
    
    // Compute sum of exponentials in parallel
    float thread_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        thread_sum += expf(logits_sample[i] - max_logit);
    }
    
    // Warp-level reduction for sum
    float sum_exp = warp_reduce_sum(thread_sum);
    
    if (tid == 0) {
        float log_sum_exp = logf(sum_exp);
        losses[sample_idx] = -(logits_sample[target] - max_logit - log_sum_exp);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be Int64 tensor");

    const int batch_size = predictions.size(0);
    const int num_classes = predictions.size(1);

    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    auto losses = torch::empty({batch_size}, predictions.options());

    const int threads_per_block = 32;
    const int blocks = batch_size;

    cross_entropy_loss_kernel_optimized<<<blocks, threads_per_block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return losses.mean();
}