#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    
    extern __shared__ float warp_sums[];
    
    float sum = 0.0f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // High-efficiency access pattern with cached reads
    for (int i = tid; i < n; i += stride) {
        float log_pred = __ldg(&log_predictions[i]);
        float target = __ldg(&targets[i]);
        sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction using shuffle instructions
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Store warp sums in shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // First warp reduces block sums
    if (warp_id == 0) {
        float block_sum = (lane_id < warps_per_block) ? warp_sums[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        
        if (lane_id == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

torch::Tensor fused_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Adaptive block sizing for different input sizes
    int block_size = 256;
    if (n > 65536) block_size = 512;
    else if (n < 8192) block_size = 128;
    
    const int max_blocks = 256;
    const int blocks = min(max_blocks, (n + block_size - 1) / block_size);
    const int shared_mem = (block_size / 32) * sizeof(float);
    
    fused_kl_div_kernel<<<blocks, block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_kl_div_forward, "Fused KLDivLoss with warp reduction and dynamic sizing (CUDA)");
}