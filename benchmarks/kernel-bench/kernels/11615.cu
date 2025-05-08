#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    const int n,
    const int chunk_offset) {
    
    extern __shared__ float sdata[];
    
    const unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 4 + tid * 4 + chunk_offset;
    const unsigned int gridSize = blockDim.x * gridDim.x * 4;
    
    float thread_sum = 0.0f;
    
    // Process elements with vector loads when possible
    while (i + 3 < n) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int idx = i + j;
            if (idx < n) {
                const float log_pred = log_predictions[idx];
                const float target = targets[idx];
                thread_sum += expf(log_pred) - target * log_pred;
            }
        }
        i += gridSize;
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Reduction within block
    if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    
    if (tid < 32) {
        // Warp-level reduction
        float warp_sum = sdata[tid];
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
    
    // Create streams for overlapped execution
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    const int threads = 256;
    const int elements_per_stream = (n + num_streams - 1) / num_streams;
    const int blocks_per_stream = min((elements_per_stream + threads * 4 - 1) / (threads * 4), 256);
    const int shared_mem = threads * sizeof(float);
    
    // Launch kernels in different streams
    for (int i = 0; i < num_streams; i++) {
        const int chunk_offset = i * elements_per_stream;
        const int chunk_size = min(elements_per_stream, n - chunk_offset);
        
        if (chunk_size > 0) {
            kl_div_kernel<<<blocks_per_stream, threads, shared_mem, streams[i]>>>(
                log_predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                output.data_ptr<float>(),
                n,
                chunk_offset
            );
        }
    }
    
    // Synchronize and cleanup streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}