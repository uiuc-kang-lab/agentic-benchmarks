#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_optimized(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int global_id = blockIdx.x * blockDim.x + tid;
    
    extern __shared__ float warp_sums[];
    
    float thread_sum = 0.0f;
    
    // Grid-stride loop for better load balancing
    int idx = global_id;
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
        idx += blockDim.x * gridDim.x;
    }

    // Warp-level reduction
    float warp_sum = warp_reduce(thread_sum);
    
    // Store warp results
    if (lane == 0) {
        warp_sums[wid] = warp_sum;
    }
    __syncthreads();

    // Final block reduction
    if (wid == 0) {
        float val = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_sums[lane] : 0.0f;
        float block_sum = warp_reduce(val);
        
        if (lane == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = BLOCK_SIZE;
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int shared_mem = warps_per_block * sizeof(float);
    
    // Use fixed grid size for better occupancy with grid-stride loop
    const int num_sms = 80;  // Adjust based on target GPU
    const int blocks = num_sms * 32;
    
    kl_div_kernel_optimized<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA optimized)");
}