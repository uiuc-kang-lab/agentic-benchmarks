#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for computing KL divergence element
__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Device function for parallel reduction
__device__ __forceinline__ void warp_reduce(volatile float* sdata, unsigned int tid) {
    if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
    if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
    if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
    if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
    if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
    if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
}

// Device function for block reduction
__device__ __forceinline__ void block_reduce(float* partial_sums, unsigned int tid) {
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    // Final warp reduction
    if (tid < 32) warp_reduce(partial_sums, tid);
}

// Main CUDA kernel with stream for overlapping
__global__ void kl_div_kernel_stream(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Get thread ID
    const unsigned int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    
    // Compute local sum
    float sum = 0.0f;
    while (idx < n) {
        sum += compute_kl_element(log_predictions[idx], targets[idx]);
        idx += stride;
    }
    
    // Store in shared memory
    partial_sums[tid] = sum;
    __syncthreads();
    
    // Perform reduction
    block_reduce(partial_sums, tid);
    
    // Write result
    if (tid == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward_stream(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);
    
    // Create a CUDA stream for overlapping
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Asynchronously launch kernel
    kl_div_kernel_stream<<<blocks, threads, shared_mem, stream>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Synchronize the stream to ensure completion
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_stream, "KL divergence forward with stream (CUDA)");
}
