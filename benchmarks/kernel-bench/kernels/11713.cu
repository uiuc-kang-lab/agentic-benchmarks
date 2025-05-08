#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void shared_memory_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    extern __shared__ float shared_data[];
    float* warp_sums = shared_data;
    float* log_pred_shared = shared_data + blockDim.x;
    float* target_shared = log_pred_shared + blockDim.x;

    float sum = 0.0f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Load data into shared memory
    for (int i = tid; i < n; i += stride) {
        log_pred_shared[threadIdx.x] = __ldg(&log_predictions[i]);
        target_shared[threadIdx.x] = __ldg(&targets[i]);
        __syncthreads();

        float log_pred = log_pred_shared[threadIdx.x];
        float target = target_shared[threadIdx.x];
        sum += expf(log_pred) - target * log_pred;
        __syncthreads();
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Store warp sum in shared memory
    if (lane_id == 0) {
        warp_sums[threadIdx.x / warp_size] = sum;
    }
    __syncthreads();

    // First warp reduces across all warps in block
    if (threadIdx.x < warp_size) {
        float block_sum = (threadIdx.x < warps_per_block) ? warp_sums[threadIdx.x] : 0.0f;
        
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        
        // Single atomic add per block
        if (threadIdx.x == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor shared_memory_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Dynamic block size selection
    int block_size = 256;
    if (n > 65536) block_size = 512;
    else if (n < 8192) block_size = 128;
    
    // Grid size calculation (max 256 blocks)
    const int blocks = min(256, (n + block_size - 1) / block_size);
    const int shared_mem = (block_size * 2 + block_size / 32) * sizeof(float);
    
    shared_memory_kl_div_kernel<<<blocks, block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_memory_kl_div_forward, "KLDivLoss with shared memory optimization (CUDA)");
}