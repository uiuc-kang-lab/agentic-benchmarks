#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void optimized_warp_vector_triplet_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int num_warps = blockDim.x >> 5;
    
    if (batch_idx >= batch_size) return;
    
    scalar_t local_dist_pos = 0;
    scalar_t local_dist_neg = 0;
    
    const int features_per_warp = (feat_size + num_warps - 1) / num_warps;
    const int warp_start = warp_id * features_per_warp;
    const int warp_end = min(warp_start + features_per_warp, feat_size);
    const int base_idx = batch_idx * feat_size + warp_start;

    if constexpr (std::is_same<scalar_t, float>::value) {
        const int vec_size = 4;
        const int aligned_start = (base_idx + 3) & ~3;
        const int aligned_end = (batch_idx * feat_size + warp_end) & ~3;
        
        for (int idx = base_idx + lane_id; idx < aligned_start; idx += 32) {
            const scalar_t a = __ldg(&anchor[idx]);
            const scalar_t p = __ldg(&positive[idx]);
            const scalar_t n = __ldg(&negative[idx]);
            
            const scalar_t d_pos = a - p;
            const scalar_t d_neg = a - n;
            
            local_dist_pos += d_pos * d_pos;
            local_dist_neg += d_neg * d_neg;
        }

        const float4* anchor_vec = reinterpret_cast<const float4*>(&anchor[aligned_start]);
        const float4* positive_vec = reinterpret_cast<const float4*>(&positive[aligned_start]);
        const float4* negative_vec = reinterpret_cast<const float4*>(&negative[aligned_start]);
        
        for (int idx = lane_id; idx < (aligned_end - aligned_start) / vec_size; idx += 32) {
            float4 a_vec = __ldg(&anchor_vec[idx]);
            float4 p_vec = __ldg(&positive_vec[idx]);
            float4 n_vec = __ldg(&negative_vec[idx]);
            
            float4 d_pos = {a_vec.x - p_vec.x, a_vec.y - p_vec.y, a_vec.z - p_vec.z, a_vec.w - p_vec.w};
            float4 d_neg = {a_vec.x - n_vec.x, a_vec.y - n_vec.y, a_vec.z - n_vec.z, a_vec.w - n_vec.w};
            
            local_dist_pos += d_pos.x * d_pos.x + d_pos.y * d_pos.y + 
                             d_pos.z * d_pos.z + d_pos.w * d_pos.w;
            local_dist_neg += d_neg.x * d_neg.x + d_neg.y * d_neg.y + 
                             d_neg.z * d_neg.z + d_neg.w * d_neg.w;
        }

        for (int idx = aligned_end + lane_id; idx < batch_idx * feat_size + warp_end; idx += 32) {
            const scalar_t a = __ldg(&anchor[idx]);
            const scalar_t p = __ldg(&positive[idx]);
            const scalar_t n = __ldg(&negative[idx]);
            
            const scalar_t d_pos = a - p;
            const scalar_t d_neg = a - n;
            
            local_dist_pos += d_pos * d_pos;
            local_dist_neg += d_neg * d_neg;
        }
    } else {
        for (int idx = base_idx + lane_id; idx < batch_idx * feat_size + warp_end; idx += 32) {
            const scalar_t a = __ldg(&anchor[idx]);
            const scalar_t p = __ldg(&positive[idx]);
            const scalar_t n = __ldg(&negative[idx]);
            
            const scalar_t d_pos = a - p;
            const scalar_t d_neg = a - n;
            
            local_dist_pos += d_pos * d_pos;
            local_dist_neg += d_neg * d_neg;
        }
    }
    
    local_dist_pos = warp_reduce(local_dist_pos);
    local_dist_neg = warp_reduce(local_dist_neg);
    
    if (lane_id == 0) {
        scalar_t warp_result = sqrt(local_dist_pos) - sqrt(local_dist_neg);
        if (warp_id == 0) {
            output[batch_idx] = max(scalar_t(0.0), warp_result + margin);
        } else {
            atomicAdd(&output[batch_idx], warp_result);
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
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "optimized_warp_vector_triplet_kernel", ([&] {
        optimized_warp_vector_triplet_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
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