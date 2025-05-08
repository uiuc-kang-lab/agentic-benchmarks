#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

__global__ void kl_div_kernel_shared(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int gid = bid * blockDim.x + tid;
    
    // Shared memory for inputs and partial sums
    extern __shared__ float shared_mem[];
    float* shared_log_pred = shared_mem;
    float* shared_targets = &shared_mem[blockDim.x];
    float* partial_sums = &shared_mem[2 * blockDim.x];
    
    float sum = 0.0f;
    
    // Process data in chunks
    const int chunks = (n + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    for (int chunk = 0; chunk < chunks; chunk++) {
        const int idx = gid + chunk * gridDim.x * blockDim.x;
        
        if (idx < n) {
            // Load chunk into shared memory
            shared_log_pred[tid] = log_predictions[idx];
            shared_targets[tid] = targets[idx];
        }
        __syncthreads();
        
        // Process elements from shared memory
        if (idx < n) {
            sum += compute_kl_element(shared_log_pred[tid], shared_targets[tid]);
        }
        __syncthreads();
    }
    
    // Store partial sum
    partial_sums[tid] = sum;
    __syncthreads();
    
    // Reduction within block
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp reduction (unrolled)
    if (tid < 32) {
        volatile float* smem = partial_sums;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }
    
    // Write result
    if (tid == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    // Shared memory for log_predictions, targets, and partial sums
    const int shared_mem = 3 * threads * sizeof(float);
    
    kl_div_kernel_shared<<<blocks, threads, shared_mem>>>(
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