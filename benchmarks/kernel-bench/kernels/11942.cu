#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void atomic_minimized_triplet_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    extern __shared__ scalar_t shared_mem[];
    scalar_t* shared_pos = shared_mem;
    scalar_t* shared_neg = &shared_mem[32];
    
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Register-based accumulation
    scalar_t local_dist_pos = 0;
    scalar_t local_dist_neg = 0;
    
    // Process 4 elements at a time using vector loads
    const int vec_size = 4;
    const int base_idx = batch_idx * feat_size;
    const int aligned_size = feat_size & ~(vec_size - 1);
    
    // Vector loads for aligned data
    for (int i = tid * vec_size; i < aligned_size; i += blockDim.x * vec_size) {
        const int idx = base_idx + i;
        float4 a4 = *reinterpret_cast<const float4*>(&anchor[idx]);
        float4 p4 = *reinterpret_cast<const float4*>(&positive[idx]);
        float4 n4 = *reinterpret_cast<const float4*>(&negative[idx]);
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const scalar_t a = reinterpret_cast<scalar_t*>(&a4)[j];
            const scalar_t p = reinterpret_cast<scalar_t*>(&p4)[j];
            const scalar_t n = reinterpret_cast<scalar_t*>(&n4)[j];
            
            const scalar_t d_pos = a - p;
            const scalar_t d_neg = a - n;
            
            local_dist_pos += d_pos * d_pos;
            local_dist_neg += d_neg * d_neg;
        }
    }
    
    // Handle remaining elements
    for (int i = aligned_size + tid; i < feat_size; i += blockDim.x) {
        const int idx = base_idx + i;
        const scalar_t a = __ldg(&anchor[idx]);
        const scalar_t p = __ldg(&positive[idx]);
        const scalar_t n = __ldg(&negative[idx]);
        
        const scalar_t d_pos = a - p;
        const scalar_t d_neg = a - n;
        
        local_dist_pos += d_pos * d_pos;
        local_dist_neg += d_neg * d_neg;
    }
    
    // Warp-level reduction
    local_dist_pos = warp_reduce_sum(local_dist_pos);
    local_dist_neg = warp_reduce_sum(local_dist_neg);
    
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Store warp results in shared memory
    if (lane_id == 0) {
        shared_pos[warp_id] = local_dist_pos;
        shared_neg[warp_id] = local_dist_neg;
    }
    
    __syncthreads();
    
    // Final reduction by first warp only
    if (warp_id == 0 && lane_id < (blockDim.x / 32)) {
        scalar_t sum_pos = shared_pos[lane_id];
        scalar_t sum_neg = shared_neg[lane_id];
        
        sum_pos = warp_reduce_sum(sum_pos);
        sum_neg = warp_reduce_sum(sum_neg);
        
        if (lane_id == 0) {
            // Single atomic write per block
            const scalar_t loss = max(scalar_t(0.0), sqrt(sum_pos) - sqrt(sum_neg) + margin);
            output[batch_idx] = loss;
        }
    }
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {
    
    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");
    
    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);
    
    auto output = torch::zeros({batch_size}, anchor.options());
    
    const int threads_per_block = 256;
    const int num_blocks = batch_size;
    const int shared_mem_size = 64 * sizeof(float); // 32 elements each for pos and neg
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "atomic_minimized_triplet_kernel", ([&] {
        atomic_minimized_triplet_kernel<scalar_t><<<num_blocks, threads_per_block, shared_mem_size>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            margin,
            batch_size,
            feat_size);
    }));
    
    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cuda, "Triplet margin loss forward (CUDA)");
}