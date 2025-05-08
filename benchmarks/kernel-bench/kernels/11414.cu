#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__
float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void compute_block_sums_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ block_sums,
    const int n) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }
    
    sum = warp_reduce_sum(sum);
    
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&block_sums[blockIdx.x], sum);
    }
}

__global__ void reduce_blocks_kernel(
    const float* __restrict__ block_sums,
    float* __restrict__ output,
    int num_blocks) {
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum += block_sums[i];
    }
    
    sum = warp_reduce_sum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // First pass: Compute block sums
    const int threads_per_block = 256;
    const int max_blocks = 256;
    const int num_blocks = min(max_blocks, (n + threads_per_block - 1) / threads_per_block);
    
    auto block_sums = torch::zeros({num_blocks}, log_predictions.options());
    
    compute_block_sums_kernel<<<num_blocks, threads_per_block>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        n
    );
    
    // Second pass: Reduce block sums
    const int reduce_threads = 256;
    reduce_blocks_kernel<<<1, reduce_threads>>>(
        block_sums.data_ptr<float>(),
        output.data_ptr<float>(),
        num_blocks
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}