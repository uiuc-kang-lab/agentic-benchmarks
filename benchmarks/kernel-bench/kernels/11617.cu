#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4
#define CHUNK_SIZE 1024

__device__ __forceinline__ float compute_kl(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int chunk_size,
    const int chunk_offset) {
    
    extern __shared__ float sdata[];
    
    const unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 4 + tid * 4 + chunk_offset;
    const unsigned int gridSize = blockDim.x * gridDim.x * 4;
    
    float thread_sum = 0.0f;
    
    // Process elements within this chunk
    const int chunk_end = chunk_offset + chunk_size;
    while (i + 3 < chunk_end) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            thread_sum += compute_kl(log_predictions[i + j], targets[i + j]);
        }
        i += gridSize;
    }
    
    // Handle remaining elements in chunk
    while (i < chunk_end) {
        thread_sum += compute_kl(log_predictions[i], targets[i]);
        i++;
    }
    
    // First level reduction - store in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Reduce within warps first
    if (tid < 32) {
        float warp_sum = 0.0f;
        #pragma unroll
        for (int i = tid; i < blockDim.x; i += 32) {
            warp_sum += sdata[i];
        }
        warp_sum = warp_reduce(warp_sum);
        
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
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    const int threads = 256;
    const int chunk_size = CHUNK_SIZE;
    const int blocks_per_chunk = min((chunk_size + threads * 4 - 1) / (threads * 4), 256);
    const int shared_mem = threads * sizeof(float);
    
    // Process data in chunks using multiple streams
    for (int chunk_start = 0; chunk_start < n; chunk_start += chunk_size * NUM_STREAMS) {
        for (int s = 0; s < NUM_STREAMS && (chunk_start + s * chunk_size) < n; s++) {
            const int current_chunk_size = min(chunk_size, n - (chunk_start + s * chunk_size));
            const int stream_offset = chunk_start + s * chunk_size;
            
            kl_div_kernel<<<blocks_per_chunk, threads, shared_mem, streams[s]>>>(
                log_predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                output.data_ptr<float>(),
                current_chunk_size,
                stream_offset
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