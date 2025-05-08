#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // 2D block configuration (16x16 = 256 threads)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int local_tid = ty * 16 + tx;
    
    // Calculate global index with 2D blocks
    const int block_offset = (blockIdx.y * gridDim.x + blockIdx.x) * 256;
    const int global_tid = block_offset + local_tid;
    
    extern __shared__ float partial_sums[];
    float thread_sum = 0.0f;
    
    // Each thread processes 4 elements with optimal striding
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        const int idx = global_tid * 4 + i;
        if(idx < n) {
            const float log_pred = log_predictions[idx];
            const float target = targets[idx];
            thread_sum += target * (logf(target) - log_pred);
        }
    }
    
    // Store in shared memory using 2D indexing
    partial_sums[local_tid] = thread_sum;
    __syncthreads();
    
    // 2D reduction in shared memory
    if(ty < 8) {
        partial_sums[local_tid] += partial_sums[local_tid + 128];
        __syncthreads();
    }
    if(ty < 4) {
        partial_sums[local_tid] += partial_sums[local_tid + 64];
        __syncthreads();
    }
    if(ty < 2) {
        partial_sums[local_tid] += partial_sums[local_tid + 32];
        __syncthreads();
    }
    
    // Final warp reduction using shuffle
    if(ty == 0) {
        float sum = partial_sums[tx];
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
        
        if(tx == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // 2D grid configuration
    dim3 threads(16, 16);  // 256 threads per block
    const int total_blocks = (n + (256 * 4) - 1) / (256 * 4);
    const int grid_y = (total_blocks + 31) / 32;  // Up to 32 blocks in x-dimension
    dim3 blocks(min(total_blocks, 32), grid_y);
    
    const int shared_mem = 256 * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
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