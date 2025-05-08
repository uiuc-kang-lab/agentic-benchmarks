#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4
#define CHUNK_SIZE 1024

__device__ __forceinline__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void pipelined_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int chunk_offset,
    const int chunk_size) {
    
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    extern __shared__ float shared[];
    float sum = 0.0f;
    
    // Process elements within this chunk
    if (gid < chunk_size) {
        const int global_idx = chunk_offset + gid;
        float log_pred = log_predictions[global_idx];
        float target = targets[global_idx];
        sum = __expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction
    sum = warp_reduce(sum);
    
    if (lane_id == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (tid < (blockDim.x / 32)) ? shared[lane_id] : 0.0f;
        sum = warp_reduce(sum);
        
        if (lane_id == 0) {
            atomicAdd(output, sum);
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
    
    const int threads = 256;
    const int shared_mem = (threads / 32) * sizeof(float);
    
    // Process data in chunks using multiple streams
    for (int chunk_start = 0; chunk_start < n; chunk_start += CHUNK_SIZE * NUM_STREAMS) {
        for (int s = 0; s < NUM_STREAMS && (chunk_start + s * CHUNK_SIZE) < n; s++) {
            const int current_chunk_start = chunk_start + s * CHUNK_SIZE;
            const int current_chunk_size = min(CHUNK_SIZE, n - current_chunk_start);
            const int blocks = (current_chunk_size + threads - 1) / threads;
            
            pipelined_kl_div_kernel<<<blocks, threads, shared_mem, streams[s]>>>(
                log_predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                output.data_ptr<float>(),
                current_chunk_start,
                current_chunk_size
            );
        }
    }
    
    // Synchronize and cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}