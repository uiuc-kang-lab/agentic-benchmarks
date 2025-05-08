#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel_optimized_syncthreads(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Shared memory for block-level reduction
    extern __shared__ float shared_sum[];
    
    // Initialize thread's accumulator
    float thread_sum = 0.0f;
    
    // Each thread processes multiple elements with grid-stride loop
    for(int idx = gid; idx < n; idx += stride) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
    }
    
    // Store thread sum in shared memory
    shared_sum[tid] = thread_sum;
    // Synchronize here to ensure all writes to shared_sum are done
    __syncthreads();
    
    // Perform reduction within block
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        // Synchronize only if there are more reductions to be done
        if (s > 1) __syncthreads();
    }
    
    // Only the first thread in each block does atomic add to global result
    if(tid == 0) {
        atomicAdd(output, shared_sum[0]);
    }
}

torch::Tensor kl_div_cuda_forward_optimized_syncthreads(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024); // Cap blocks for better occupancy
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel_optimized_syncthreads<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_optimized_syncthreads, "KL divergence forward (CUDA)");
}