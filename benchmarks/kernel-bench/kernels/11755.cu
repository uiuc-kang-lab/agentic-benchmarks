#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for KL divergence calculation 
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Constants for warp operations
    const unsigned int warp_size = 32;
    const unsigned int lane_id = threadIdx.x % warp_size;
    const unsigned int warp_id = threadIdx.x / warp_size;
    const unsigned int warps_per_block = blockDim.x / warp_size;
    
    // Shared memory only needed for final warp reduction
    extern __shared__ float warp_results[];
    
    float sum = 0.0f;
    
    // Calculate global index and stride
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process elements with stride pattern
    for (int i = tid; i < n; i += stride) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction across warps (only first warp)
    if (warp_id == 0) {
        // Load warp result or zero if lane is beyond number of warps
        float warp_sum = (lane_id < warps_per_block) ? warp_results[lane_id] : 0.0f;
        
        // Final warp-level reduction
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // First thread adds to global output
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

// Host function to manage streams and memory transfers
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize launch configuration
    const int threads_per_block = 256; // Multiple of warp size (32)
    const int num_warps = threads_per_block / 32;
    const int blocks = min(256, (n + threads_per_block - 1) / threads_per_block);
    const int shared_mem = num_warps * sizeof(float);
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Asynchronous memory copy to device
    float* d_log_predictions;
    float* d_targets;
    float* d_output;
    cudaMalloc((void**)&d_log_predictions, n * sizeof(float));
    cudaMalloc((void**)&d_targets, n * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));
    cudaMemcpyAsync(d_log_predictions, log_predictions.data_ptr<float>(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_targets, targets.data_ptr<float>(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_output, 0, sizeof(float), stream);

    // Launch kernel
    kl_div_kernel<<<blocks, threads_per_block, shared_mem, stream>>>(
        d_log_predictions,
        d_targets,
        d_output,
        n
    );

    // Asynchronous copy result back to host
    cudaMemcpyAsync(output.data_ptr<float>(), d_output, sizeof(float), cudaMemcpyDeviceToHost, stream);

    // Synchronize stream
    cudaStreamSynchronize(stream);

    // Free device memory
    cudaFree(d_log_predictions);
    cudaFree(d_targets);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward with streams (CUDA)");
}