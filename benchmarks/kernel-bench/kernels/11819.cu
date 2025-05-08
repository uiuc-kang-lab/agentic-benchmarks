#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int NUM_STREAMS = 4;

__device__ inline float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kl_div_kernel_stream(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int chunk_size,
    const int offset) {
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    __shared__ float shared[32];
    float sum = 0.0f;
    
    // Process chunk with coalesced memory access
    for (int idx = gid; idx < chunk_size; idx += stride) {
        const int global_idx = idx + offset;
        const float log_pred = log_predictions[global_idx];
        const float target = targets[global_idx];
        sum += __expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction
    sum = warp_reduce(sum);
    
    // Block-level reduction
    if (tid % 32 == 0) {
        shared[tid / 32] = sum;
    }
    __syncthreads();
    
    if (tid < 32) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[tid] : 0.0f;
        sum = warp_reduce(sum);
        if (tid == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward_streamed(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    const int chunk_size = (n + NUM_STREAMS - 1) / NUM_STREAMS;
    const int threads = 256;
    const int blocks_per_chunk = min((chunk_size + threads - 1) / threads, 1024);
    
    // Launch kernels in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        const int offset = i * chunk_size;
        const int current_chunk_size = min(chunk_size, n - offset);
        
        if (current_chunk_size > 0) {
            kl_div_kernel_stream<<<blocks_per_chunk, threads, 0, streams[i]>>>(
                log_predictions.data_ptr<float>() + offset,
                targets.data_ptr<float>() + offset,
                output.data_ptr<float>(),
                current_chunk_size,
                offset
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
    m.def("forward", &kl_div_cuda_forward_streamed, "Streamed KL divergence forward (CUDA)");
}