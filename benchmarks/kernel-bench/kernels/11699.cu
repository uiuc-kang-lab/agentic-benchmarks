#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void adaptive_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = BLOCK_SIZE / warp_size;
    
    extern __shared__ float warp_results[];
    
    float sum = 0.0f;
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int stride = BLOCK_SIZE * gridDim.x;
    
    // Grid-stride loop with vectorized loads where possible
    #pragma unroll 4
    for (int i = tid; i < n; i += stride) {
        float log_pred = __ldg(&log_predictions[i]);
        float target = __ldg(&targets[i]);
        sum += __expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        warp_results[warp_id] = sum;
    }
    
    __syncthreads();
    
    if (warp_id == 0) {
        float warp_sum = (lane_id < warps_per_block) ? warp_results[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor adaptive_block_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // More granular block size selection based on input size
    const int max_blocks = 256;
    int blocks;
    void* kernel_ptr;
    int shared_mem;
    
    if (n <= 2048) {
        const int block_size = 32;
        blocks = min(max_blocks, (n + block_size - 1) / block_size);
        shared_mem = (block_size / 32) * sizeof(float);
        kernel_ptr = (void*)adaptive_kl_div_kernel<32>;
    } else if (n <= 4096) {
        const int block_size = 64;
        blocks = min(max_blocks, (n + block_size - 1) / block_size);
        shared_mem = (block_size / 32) * sizeof(float);
        kernel_ptr = (void*)adaptive_kl_div_kernel<64>;
    } else if (n <= 16384) {
        const int block_size = 128;
        blocks = min(max_blocks, (n + block_size - 1) / block_size);
        shared_mem = (block_size / 32) * sizeof(float);
        kernel_ptr = (void*)adaptive_kl_div_kernel<128>;
    } else if (n <= 65536) {
        const int block_size = 256;
        blocks = min(max_blocks, (n + block_size - 1) / block_size);
        shared_mem = (block_size / 32) * sizeof(float);
        kernel_ptr = (void*)adaptive_kl_div_kernel<256>;
    } else {
        const int block_size = 512;
        blocks = min(max_blocks, (n + block_size - 1) / block_size);
        shared_mem = (block_size / 32) * sizeof(float);
        kernel_ptr = (void*)adaptive_kl_div_kernel<512>;
    }
    
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel_ptr);
    
    // Launch kernel with selected configuration
    void (*kernel)(const float*, const float*, float*, const int) = 
        (void (*)(const float*, const float*, float*, const int))kernel_ptr;
    
    kernel<<<blocks, attr.maxThreadsPerBlock, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &adaptive_block_kl_div_forward, "KLDivLoss with adaptive block size (CUDA)");
}