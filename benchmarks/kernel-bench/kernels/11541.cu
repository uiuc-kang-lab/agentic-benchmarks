#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void partial_reduction_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ partial_results,
    const int n) {
    
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int grid_stride = gridDim.x * blockDim.x;
    
    extern __shared__ float shared_mem[];
    
    float thread_sum = 0.0f;
    
    // Grid stride loop for coalesced access
    for (int idx = gid; idx < n; idx += grid_stride) {
        thread_sum += compute_kl_div(log_predictions[idx], targets[idx]);
    }
    
    // Warp reduction
    thread_sum = warp_reduce_sum(thread_sum);
    
    if (lane_id == 0) {
        shared_mem[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction within block
    if (warp_id == 0) {
        float sum = (tid < blockDim.x/32) ? shared_mem[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        
        if (lane_id == 0) {
            partial_results[blockIdx.x] = sum;
        }
    }
}

__global__ void final_reduction_kernel(
    const float* __restrict__ partial_results,
    float* __restrict__ output,
    const int num_blocks) {
    
    extern __shared__ float shared_mem[];
    const int tid = threadIdx.x;
    
    float sum = 0.0f;
    
    // Load and sum partial results
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial_results[i];
    }
    
    // Warp reduction
    sum = warp_reduce_sum(sum);
    
    if (tid % 32 == 0) {
        shared_mem[tid/32] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (tid < 32) {
        float final_sum = (tid < (blockDim.x + 31)/32) ? shared_mem[tid] : 0.0f;
        final_sum = warp_reduce_sum(final_sum);
        
        if (tid == 0) {
            *output = final_sum;
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Launch parameters
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = (threads/32) * sizeof(float);
    
    // Allocate storage for partial results
    auto partial_results = torch::empty({blocks}, log_predictions.options());
    auto output = torch::zeros({1}, log_predictions.options());
    
    // First kernel: partial reductions
    partial_reduction_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_results.data_ptr<float>(),
        n
    );
    
    // Second kernel: final reduction
    final_reduction_kernel<<<1, threads, shared_mem>>>(
        partial_results.data_ptr<float>(),
        output.data_ptr<float>(),
        blocks
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}