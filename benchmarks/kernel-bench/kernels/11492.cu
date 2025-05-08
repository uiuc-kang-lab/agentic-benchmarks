#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const unsigned int tid = threadIdx.x;
    const unsigned int idx_base = blockIdx.x * blockDim.x * 4 + tid;
    extern __shared__ float partial_sums[];
    
    // Process 4 elements per thread using registers
    float thread_sum = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned int idx = idx_base + i * blockDim.x;
        if (idx < n) {
            const float log_pred = log_predictions[idx];
            const float target = targets[idx];
            thread_sum += expf(log_pred) - target * log_pred;
        }
    }
    
    // Store in shared memory
    partial_sums[tid] = thread_sum;
    __syncthreads();
    
    // Warp-uniform reduction
    #pragma unroll
    for (int offset = blockDim.x/2; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            partial_sums[tid] += partial_sums[tid + offset];
        }
        __syncthreads();
    }
    
    // Final warp reduction - all threads in warp participate
    if (tid < 32) {
        float warp_sum = partial_sums[tid];
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
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
    
    const int threads = 256;
    const int blocks = (n + (threads * 4) - 1) / (threads * 4);
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
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