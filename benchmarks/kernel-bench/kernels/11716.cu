#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for KL divergence calculation with shared memory optimization
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Shared memory for input data caching
    extern __shared__ float shared_data[];
    float* shared_log_pred = shared_data;
    float* shared_targets = &shared_data[blockDim.x];
    float* partial_sums = &shared_data[2 * blockDim.x];
    
    const int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    float sum = 0.0f;
    
    // Process data in chunks to maximize shared memory usage
    for (int chunk_start = blockIdx.x * blockDim.x;
         chunk_start < n;
         chunk_start += blockDim.x * gridDim.x) {
        
        // Load chunk into shared memory
        if (chunk_start + tid < n) {
            shared_log_pred[tid] = log_predictions[chunk_start + tid];
            shared_targets[tid] = targets[chunk_start + tid];
        }
        __syncthreads();
        
        // Process elements in current chunk
        const int chunk_size = min(blockDim.x, n - chunk_start);
        if (idx < n) {
            float log_pred = shared_log_pred[tid];
            float target = shared_targets[tid];
            sum += expf(log_pred) - target * log_pred;
            idx += blockDim.x * gridDim.x;
        }
        __syncthreads();
    }
    
    // Store partial sum
    partial_sums[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
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