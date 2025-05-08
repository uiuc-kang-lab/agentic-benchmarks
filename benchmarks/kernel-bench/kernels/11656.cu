#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for computing KL divergence element
__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Device function for warp reduction
__device__ __forceinline__ void warp_reduce(volatile float* sdata, unsigned int tid) {
    if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
    if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
    if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
    if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
    if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
    if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
}

// Main CUDA kernel with stream support
__global__ void kl_div_kernel_streamed(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int chunk_size,
    const int chunk_offset) {
    
    const unsigned int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid + chunk_offset;
    const int end_idx = chunk_offset + chunk_size;
    
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    if (idx < end_idx) {
        sum = compute_kl_element(log_predictions[idx], targets[idx]);
    }
    
    partial_sums[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp reduction
    if (tid < 32) warp_reduce(partial_sums, tid);
    
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
    const int num_streams = 4; // Number of concurrent streams
    const int chunk_size = (n + num_streams - 1) / num_streams;
    const int blocks_per_chunk = (chunk_size + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    // Create CUDA streams
    cudaStream_t streams[4];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Launch kernels in different streams
    for (int i = 0; i < num_streams; i++) {
        const int current_chunk_size = min(chunk_size, n - i * chunk_size);
        if (current_chunk_size <= 0) break;
        
        const int current_blocks = min(blocks_per_chunk, 
                                     (current_chunk_size + threads - 1) / threads);
        
        kl_div_kernel_streamed<<<current_blocks, threads, shared_mem, streams[i]>>>(
            log_predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            current_chunk_size,
            i * chunk_size
        );
    }
    
    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}