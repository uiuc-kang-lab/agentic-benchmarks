#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp-level reduction using shuffle instructions
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel with warp-level reduction
template <typename scalar_t>
__global__ void triplet_margin_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Register-based accumulation
    scalar_t local_dist_pos = 0;
    scalar_t local_dist_neg = 0;
    
    // Each thread processes multiple features
    const int base_idx = batch_idx * feat_size;
    
    #pragma unroll 4
    for (int feat_idx = tid; feat_idx < feat_size; feat_idx += blockDim.x) {
        const int idx = base_idx + feat_idx;
        const scalar_t a = anchor[idx];
        const scalar_t p = positive[idx];
        const scalar_t n = negative[idx];
        
        const scalar_t d_pos = a - p;
        const scalar_t d_neg = a - n;
        
        local_dist_pos += d_pos * d_pos;
        local_dist_neg += d_neg * d_neg;
    }
    
    // Warp-level reduction
    local_dist_pos = warp_reduce_sum(local_dist_pos);
    local_dist_neg = warp_reduce_sum(local_dist_neg);
    
    // Final reduction by first thread in the warp
    if (tid % 32 == 0) {
        const scalar_t final_dist_pos = sqrt(local_dist_pos);
        const scalar_t final_dist_neg = sqrt(local_dist_neg);
        output[batch_idx] = max(scalar_t(0.0), final_dist_pos - final_dist_neg + margin);
    }
}

// CUDA entry point
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
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel", ([&] {
        triplet_margin_loss_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
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