#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int VEC_SIZE = 4;
constexpr int ELEMENTS_PER_THREAD = 16;

__global__ void balanced_kl_kernel(
    const float* __restrict__ log_preds,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int n) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int elements_per_block = (n + gridDim.x - 1) / gridDim.x;
    const int block_start = blockIdx.x * elements_per_block;
    const int block_end = min(block_start + elements_per_block, n);
    
    float sum = 0.0f;
    
    // Process vectorized elements first
    const int vec_start = block_start + threadIdx.x * VEC_SIZE;
    const int vec_stride = blockDim.x * VEC_SIZE;
    
    for (int i = vec_start; i < block_end; i += vec_stride) {
        if (i + VEC_SIZE <= block_end) {
            float4 log_vec = *reinterpret_cast<const float4*>(log_preds + i);
            float4 tgt_vec = *reinterpret_cast<const float4*>(targets + i);
            
            sum += expf(log_vec.x) - tgt_vec.x * log_vec.x;
            sum += expf(log_vec.y) - tgt_vec.y * log_vec.y;
            sum += expf(log_vec.z) - tgt_vec.z * log_vec.z;
            sum += expf(log_vec.w) - tgt_vec.w * log_vec.w;
        }
    }
    
    // Process remaining elements
    const int scalar_start = block_start + threadIdx.x;
    const int scalar_stride = blockDim.x;
    
    for (int i = scalar_start; i < block_end; i += scalar_stride) {
        if (i < n && (i - block_start) >= (block_end - block_start) / VEC_SIZE * VEC_SIZE) {
            sum += expf(log_preds[i]) - targets[i] * log_preds[i];
        }
    }
    
    // Warp-level reduction
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    // Block-level reduction
    __shared__ float shared[WARP_SIZE];
    if (threadIdx.x % WARP_SIZE == 0)
        shared[threadIdx.x/WARP_SIZE] = sum;
    __syncthreads();
    
    if (threadIdx.x < WARP_SIZE) {
        float val = (threadIdx.x < blockDim.x/WARP_SIZE) ? shared[threadIdx.x] : 0.0f;
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        
        if (threadIdx.x == 0)
            atomicAdd(output, val);
    }
}

torch::Tensor balanced_kl_forward(
    torch::Tensor log_preds,
    torch::Tensor targets) {
    
    const int n = log_preds.numel();
    auto output = torch::zeros({1}, log_preds.options());
    
    // H100-optimized launch config
    const int threads = 128;
    const int sm_count = 144;  // H100 SMs
    const int blocks = min(sm_count * 4, (n + threads * ELEMENTS_PER_THREAD - 1) / (threads * ELEMENTS_PER_THREAD));
    
    balanced_kl_kernel<<<blocks, threads, WARP_SIZE * sizeof(float)>>>(
        log_preds.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &balanced_kl_forward, "Balanced workload KL divergence (CUDA)");
}