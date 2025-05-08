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
    extern __shared__ float warp_results[];
    
    float sum = 0.0f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    // Grid-stride loop with read-only cache
    for (int i = tid; i < n; i += stride) {
        sum += expf(__ldg(&log_predictions[i])) - __ldg(&targets[i]) * __ldg(&log_predictions[i]);
    }

    // Warp-level reduction
    for (int offset = warp_size/2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane_id == 0)
        warp_results[warp_id] = sum;
    __syncthreads();

    // Block-level reduction
    if (warp_id == 0) {
        float block_sum = (lane_id < warps_per_block) ? warp_results[lane_id] : 0.0f;
        for (int offset = warp_size/2; offset > 0; offset /= 2)
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        
        if (lane_id == 0)
            atomicAdd(output, block_sum);
    }
}

torch::Tensor fused_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Dynamic block selection optimized for modern GPUs
    int block_size = n >= 131072 ? 512 : 
                     n >= 32768  ? 256 : 128;
    
    int max_blocks = std::min(256, (n + block_size - 1) / block_size);
    int shared_mem = (block_size/32) * sizeof(float);
    
    fused_kl_div_kernel<<<max_blocks, block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_kl_div_forward, "Optimized KLDivLoss with dynamic blocks & warp reductions (CUDA)");
}