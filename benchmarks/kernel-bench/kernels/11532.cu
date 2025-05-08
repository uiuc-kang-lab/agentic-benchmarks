#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel function for KL divergence calculation
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_threads = blockDim.x;
    const int warp_size = 32;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    partial_sums[tid] = 0.0f;
    
    // Process elements with stride of complete warps
    for (int i = bid * num_threads + tid; i < n; i += gridDim.x * num_threads) {
        if (i < n) {
            float log_pred = log_predictions[i];
            float target = targets[i];
            partial_sums[tid] += expf(log_pred) - target * log_pred;
        }
    }
    __syncthreads();
    
    // Warp-level reduction first
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial_sums[tid] += partial_sums[tid + offset];
        }
        __syncthreads();
    }
    
    // Block-level reduction using shared memory
    if (tid == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < num_threads; i += warp_size) {
            block_sum += partial_sums[i];
        }
        atomicAdd(output, block_sum);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters optimized for memory coalescing
    const int threads = 256; // Multiple of warp size (32)
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    // Create CUDA stream for asynchronous execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Launch kernel on specified stream
    kl_div_kernel<<<blocks, threads, shared_mem, stream>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    // Synchronize stream to ensure completion
    cudaStreamSynchronize(stream);
    
    // Destroy stream after use
    cudaStreamDestroy(stream);
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}