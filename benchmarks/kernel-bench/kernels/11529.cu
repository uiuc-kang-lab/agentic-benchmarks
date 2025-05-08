#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for KL divergence computation
__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

// Warp-level reduction using shuffle instructions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Shared memory for partial sums (one float per warp)
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int grid_stride = blockDim.x * gridDim.x;
    
    // Thread-local accumulator
    float sum = 0.0f;
    
    // Grid-stride loop for coalesced memory access
    #pragma unroll 4
    for (int idx = gid; idx < n; idx += grid_stride) {
        sum += compute_kl_div(log_predictions[idx], targets[idx]);
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    
    // Write warp results to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps (done by first warp)
    if (warp_id == 0 && lane_id < (blockDim.x / 32)) {
        float warp_sum = shared_mem[lane_id];
        warp_sum = warp_reduce_sum(warp_sum);
        
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize launch parameters
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = warps_per_block * sizeof(float);
    
    optimized_kl_div_kernel<<<blocks, threads, shared_mem>>>(
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