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
__global__ void pure_warp_triplet_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    const int warp_id = tid >> 5;  // tid / 32
    const int lane_id = tid & 31;  // tid % 32
    const int num_warps = blockDim.x >> 5;
    
    if (batch_idx >= batch_size) return;
    
    // Register-based accumulation
    scalar_t local_dist_pos = 0;
    scalar_t local_dist_neg = 0;
    
    // Each warp processes a contiguous chunk of the feature vector
    const int features_per_warp = (feat_size + num_warps - 1) / num_warps;
    const int warp_start = warp_id * features_per_warp;
    const int warp_end = min(warp_start + features_per_warp, feat_size);
    
    // Process 4 elements per iteration for better memory coalescing
    const int base_idx = batch_idx * feat_size + warp_start;
    
    // Vector loads for better memory coalescing
    for (int idx = base_idx + lane_id; idx < batch_idx * feat_size + warp_end; idx += 32) {
        const scalar_t a = __ldg(&anchor[idx]);
        const scalar_t p = __ldg(&positive[idx]);
        const scalar_t n = __ldg(&negative[idx]);
        
        const scalar_t d_pos = a - p;
        const scalar_t d_neg = a - n;
        
        local_dist_pos += d_pos * d_pos;
        local_dist_neg += d_neg * d_neg;
    }
    
    // Warp-level reduction
    local_dist_pos = warp_reduce(local_dist_pos);
    local_dist_neg = warp_reduce(local_dist_neg);
    
    if (lane_id == 0) {
        if (warp_id == 0) {
            // First warp directly writes the result
            output[batch_idx] = max(scalar_t(0.0), 
                sqrt(local_dist_pos) - sqrt(local_dist_neg) + margin);
        } else {
            // Other warps contribute their partial sums
            atomicAdd(&output[batch_idx], sqrt(local_dist_pos) - sqrt(local_dist_neg));
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
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "pure_warp_triplet_kernel", ([&] {
        pure_warp_triplet_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
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