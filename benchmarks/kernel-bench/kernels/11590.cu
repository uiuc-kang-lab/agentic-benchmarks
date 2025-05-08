#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute KL divergence for a single element
__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Device function for warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel for KL divergence with optimally distributed workloads
__global__ void kl_div_kernel_balanced_workload(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    // Calculate global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the thread's range
    int elements_per_thread = (n + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    int start_idx = tid * elements_per_thread;
    int end_idx = min(start_idx + elements_per_thread, n);

    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    float sum = 0.0f;
    
    // Calculate KL divergence for this thread's range
    for (int i = start_idx; i < end_idx; i++) {
        sum += compute_kl_div(log_predictions[i], targets[i]);
    }
    
    // Store in shared memory
    partial_sums[threadIdx.x] = sum;
    __syncthreads();
    
    // Perform block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Use atomic operation only once per block
    if (threadIdx.x == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward_balanced_workload(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel_balanced_workload<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_balanced_workload, "KL divergence forward with balanced workload (CUDA)");
}
