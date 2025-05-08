#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n,
    float* __restrict__ block_results) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = gridDim.x;
    
    // Shared memory for block-level reduction
    extern __shared__ float shared_data[];
    
    float thread_sum = 0.0f;
    
    // Calculate number of elements per block
    const int items_per_grid = (n + grid_size - 1) / grid_size;
    const int block_start = bid * items_per_grid;
    const int block_end = min(block_start + items_per_grid, n);
    
    // Each thread processes multiple elements within its block's range
    for (int i = block_start + tid; i < block_end; i += block_size) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        thread_sum += expf(log_pred) - target * log_pred;
    }
    
    // Store in shared memory
    shared_data[tid] = thread_sum;
    __syncthreads();
    
    // Block-level reduction
    #pragma unroll
    for (int stride = block_size/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Store block result
    if (tid == 0) {
        block_results[bid] = shared_data[0];
    }
    
    // Final reduction by block 0
    if (bid == 0 && tid == 0) {
        float final_sum = 0.0f;
        for (int i = 0; i < grid_size; i++) {
            final_sum += block_results[i];
        }
        *output = final_sum;
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Create output tensor and temporary storage for block results
    auto output = torch::zeros({1}, log_predictions.options());
    auto block_results = torch::empty({80}, log_predictions.options()); // H100 has 80 SMs
    
    // Launch parameters optimized for H100
    const int threads = 256;
    const int blocks = 80; // Match SM count for H100
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        block_results.data_ptr<float>()
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}