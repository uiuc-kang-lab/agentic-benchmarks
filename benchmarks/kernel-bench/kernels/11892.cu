#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_shared_memory(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int global_thread_id = blockIdx.x * blockDim.x + tid;
    extern __shared__ float shared_data[];
    
    float thread_sum = 0.0f;
    
    // Grid-stride loop
    for (int idx = global_thread_id; idx < n; idx += blockDim.x * gridDim.x) {
        const float log_pred = log_predictions[idx];
        const float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
    }
    
    // Store thread results in shared memory
    shared_data[tid] = thread_sum;
    __syncthreads();
    
    // Intra-block reduction using shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Final reduction using warp-level primitives
    if (tid < WARP_SIZE) {
        float block_sum = (tid == 0) ? shared_data[0] : 0.0f;
        block_sum = warp_reduce(block_sum);
        if (tid == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = BLOCK_SIZE;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel_shared_memory<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA shared memory optimized)");
}