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

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    extern __shared__ float shared_data[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    
    // Grid-stride loop
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
        idx += stride;
    }

    // Each thread puts its local sum into shared memory
    shared_data[tid] = sum;
    __syncthreads();

    // Intra-block reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(output, shared_data[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Calculate optimal thread/block configuration
    const int threads_per_block = 256;
    const int max_blocks = 256;
    const int num_blocks = min(max_blocks, (n + threads_per_block - 1) / threads_per_block);
    
    auto output = torch::zeros({1}, log_predictions.options());
    
    kl_div_kernel<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(
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