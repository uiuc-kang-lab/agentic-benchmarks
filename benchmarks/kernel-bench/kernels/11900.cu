#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4
#define NUM_STREAMS 4

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_streamed(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int chunk_size,
    const int chunk_offset) {
    
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int global_thread_id = blockIdx.x * blockDim.x + tid;
    
    extern __shared__ float warp_results[];
    
    float thread_sum = 0.0f;
    
    // Process elements within this stream's chunk
    const int start_idx = chunk_offset + global_thread_id * ELEMENTS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int idx = start_idx + i;
        if (idx < chunk_offset + chunk_size) {
            const float log_pred = log_predictions[idx];
            const float target = targets[idx];
            thread_sum += expf(log_pred) - target * log_pred;
        }
    }
    
    // Warp-level reduction
    thread_sum = warp_reduce(thread_sum);
    
    if (lane == 0) {
        warp_results[wid] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (wid == 0) {
        float warp_sum = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_results[lane] : 0.0f;
        warp_sum = warp_reduce(warp_sum);
        
        if (lane == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Calculate chunk size for each stream
    const int chunk_size = (n + NUM_STREAMS - 1) / NUM_STREAMS;
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int shared_mem = warps_per_block * sizeof(float);
    
    // Launch kernels in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        const int chunk_offset = i * chunk_size;
        const int current_chunk_size = min(chunk_size, n - chunk_offset);
        
        if (current_chunk_size <= 0) continue;
        
        const int blocks_needed = (current_chunk_size + BLOCK_SIZE * ELEMENTS_PER_THREAD - 1) 
                                / (BLOCK_SIZE * ELEMENTS_PER_THREAD);
        
        kl_div_kernel_streamed<<<blocks_needed, BLOCK_SIZE, shared_mem, streams[i]>>>(
            log_predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            current_chunk_size,
            chunk_offset
        );
    }
    
    // Synchronize and cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA streamed)");
}