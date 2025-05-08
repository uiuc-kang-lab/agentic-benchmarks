#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int VECTOR_SIZE = 4;
constexpr int THREADS_PER_BLOCK = 512;
// H100 has 132 SMs, target 2 blocks/SM for occupancy
constexpr int MAX_GRID_DIM = 264;

__global__ void coalesced_kl_kernel(
    const float* __restrict__ log_p,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int total_elements) {
    
    const int vector_total = total_elements / VECTOR_SIZE;
    const int total_warps = (THREADS_PER_BLOCK + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float block_results[total_warps];
    
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    float thread_sum = 0.0f;

    // Aligned vector loop (each thread handles VECTOR_SIZE consecutive elements)
    for(int vec_idx = global_tid; vec_idx < vector_total; vec_idx += gridDim.x * blockDim.x) {
        const float4 log_vec = *reinterpret_cast<const float4*>(log_p + vec_idx*VECTOR_SIZE);
        const float4 tgt_vec = *reinterpret_cast<const float4*>(targets + vec_idx*VECTOR_SIZE);
        
        thread_sum += expf(log_vec.x) - tgt_vec.x * log_vec.x;
        thread_sum += expf(log_vec.y) - tgt_vec.y * log_vec.y;
        thread_sum += expf(log_vec.z) - tgt_vec.z * log_vec.z;
        thread_sum += expf(log_vec.w) - tgt_vec.w * log_vec.w;
    }

    // Handle remaining elements (non-vectorized loop)
    const int scalar_start = vector_total * VECTOR_SIZE + global_tid;
    for(int idx = scalar_start; idx < total_elements; idx += gridDim.x * blockDim.x) {
        const float log_val = log_p[idx];
        const float tgt_val = targets[idx];
        thread_sum += expf(log_val) - tgt_val * log_val;
    }

    // Reduction within warp
    float warp_sum = thread_sum;
    for(int meta_offset = WARP_SIZE/2; meta_offset > 0; meta_offset >>= 1)
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, meta_offset);
    
    // Single atomic per warp
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    if(lane_id == 0)
        atomicAdd(&block_results[warp_id], warp_sum);
    
    __syncthreads();

    // Final block reduction
    if(threadIdx.x == 0) {
        float block_total = 0;
        for(int i = 0; i < total_warps; i++) 
            block_total += block_results[i];
        
        atomicAdd(output, block_total);
    }
}

torch::Tensor optimized_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Calculate grid size based on compute capacity
    const int blocks = min(
        MAX_GRID_DIM, 
        (n / VECTOR_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
    );
    
    coalesced_kl_kernel<<<blocks, THREADS_PER_BLOCK, blocks*sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_kl_forward, "KLDivLoss with grid-stride coalesced access (CUDA)");
}