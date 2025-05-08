#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel_2d(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int rows,
    const int cols) {
    
    // 2D thread indexing
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Process elements using 2D indexing
    if (row < rows && col < cols) {
        const int idx = row * cols + col;
        
        // Use vectorized loads if possible
        if (col + 4 <= cols) {
            float4 log_pred = *reinterpret_cast<const float4*>(&log_predictions[idx]);
            float4 target = *reinterpret_cast<const float4*>(&targets[idx]);
            
            // Process vector elements
            sum += expf(log_pred.x) - target.x * log_pred.x;
            sum += expf(log_pred.y) - target.y * log_pred.y;
            sum += expf(log_pred.z) - target.z * log_pred.z;
            sum += expf(log_pred.w) - target.w * log_pred.w;
        } else {
            float log_pred = log_predictions[idx];
            float target = targets[idx];
            sum += expf(log_pred) - target * log_pred;
        }
    }
    
    // Store in shared memory
    partial_sums[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
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
    
    // Get tensor dimensions
    const int rows = log_predictions.size(0);
    const int cols = log_predictions.size(1);
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Launch parameters optimized for 2D indexing
    dim3 threads(16, 16);  // 256 threads per block in 16x16 configuration
    dim3 blocks(
        (cols + threads.x - 1) / threads.x,
        (rows + threads.y - 1) / threads.y
    );
    const int shared_mem = threads.x * threads.y * sizeof(float);
    
    // Launch kernel
    kl_div_kernel_2d<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );
    
    return output / static_cast<float>(rows * cols);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}