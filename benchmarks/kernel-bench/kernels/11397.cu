#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {
    
    // Get thread and block indices
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_threads = blockDim.x;
    const int num_blocks = gridDim.x;
    
    // Calculate total stride
    const int stride = num_threads * num_blocks;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    partial_sums[tid] = 0.0f;
    
    // Each thread processes multiple elements with stride
    for(int i = bid * num_threads + tid; i < n; i += stride) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        partial_sums[tid] += expf(log_pred) - target * log_pred;
    }
    
    __syncthreads();
    
    // Parallel reduction in shared memory
    for(int s = num_threads/2; s > 0; s >>= 1) {
        if(tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if(tid == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize thread and block count based on data size
    const int threads = 256;
    const int blocks = min(256, (n + threads - 1) / threads);
    const int shared_mem = threads * sizeof(float);
    
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