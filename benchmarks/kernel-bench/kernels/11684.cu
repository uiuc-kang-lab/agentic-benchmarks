#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void shared_memory_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    extern __shared__ float shared_data[];
    float* log_pred_shared = shared_data;
    float* target_shared = shared_data + blockDim.x;
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    
    for (int i = idx; i < n; i += stride) {
        // Load data into shared memory
        if (i < n) {
            log_pred_shared[tid] = log_predictions[i];
            target_shared[tid] = targets[i];
        }
        __syncthreads();
        
        // Compute using shared memory
        if (i < n) {
            float log_pred = log_pred_shared[tid];
            float target = target_shared[tid];
            sum += expf(log_pred) - target * log_pred;
        }
        __syncthreads();
    }
    
    // Store partial sum in shared memory
    shared_data[tid] = sum;
    __syncthreads();
    
    // Reduce within block
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_data[tid] += shared_data[tid + offset];
        }
        __syncthreads();
    }
    
    // Write block's result to global memory
    if (tid == 0) {
        atomicAdd(output, shared_data[0]);
    }
}

torch::Tensor shared_memory_kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = 2 * threads * sizeof(float);
    
    shared_memory_kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_memory_kl_div_cuda_forward, "KL divergence forward with shared memory optimization (CUDA)");
}