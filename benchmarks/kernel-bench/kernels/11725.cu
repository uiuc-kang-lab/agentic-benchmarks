#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for KL divergence calculation with shared memory and reduced synchronization
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Allocate shared memory for input data caching and partial sums
    extern __shared__ float shared_data[];
    float* shared_log_pred = shared_data;
    float* shared_targets = &shared_data[blockDim.x];
    float* partial_sums = &shared_data[2 * blockDim.x];
    
    float sum = 0.0f;
    const int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load data into shared memory in chunks optimally
    for (int chunk_start = blockIdx.x * blockDim.x;
         chunk_start < n;
         chunk_start += blockDim.x * gridDim.x) {
        
        // Load chunk into shared memory if within bounds and only sync when necessary
        if (chunk_start + tid < n) {
            shared_log_pred[tid] = log_predictions[chunk_start + tid];
            shared_targets[tid] = targets[chunk_start + tid];
        }
        __syncthreads();  // Sync before computation to ensure data loading is complete
        
        const int chunk_size = min(blockDim.x, n - chunk_start);
        if (tid < chunk_size) {  // Only valid threads participate
            float log_pred = shared_log_pred[tid];
            float target = shared_targets[tid];
            sum += expf(log_pred) - target * log_pred;
        }
        __syncthreads();  // Sync before the next iteration to avoid race conditions
    }
    
    // Reduction in shared memory
    partial_sums[tid] = sum;
    __syncthreads();  // Ensure all partial sums are written
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    // Atomic add the result of each block to the global output
    if (tid == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    // Shared memory size: space for input caching and partial sums
    const int shared_mem = (2 * threads + threads) * sizeof(float);
    
    // Launch kernel
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
