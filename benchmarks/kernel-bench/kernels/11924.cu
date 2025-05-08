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
__global__ void triplet_margin_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int sample_idx = warp_id;
    
    if (sample_idx >= batch_size) return;
    
    scalar_t dist_pos = 0;
    scalar_t dist_neg = 0;
    
    // Vector loading for 4 elements at a time when possible
    const int vec_size = 4;
    const int vec_feat_size = feat_size / vec_size;
    const int base_idx = sample_idx * feat_size;
    
    // Vector loads for aligned portions
    #pragma unroll 4
    for (int i = lane_id; i < vec_feat_size; i += 32) {
        const int vec_idx = base_idx + i * vec_size;
        float4 a4 = *reinterpret_cast<const float4*>(&anchor[vec_idx]);
        float4 p4 = *reinterpret_cast<const float4*>(&positive[vec_idx]);
        float4 n4 = *reinterpret_cast<const float4*>(&negative[vec_idx]);
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const scalar_t a = reinterpret_cast<scalar_t*>(&a4)[j];
            const scalar_t p = reinterpret_cast<scalar_t*>(&p4)[j];
            const scalar_t n = reinterpret_cast<scalar_t*>(&n4)[j];
            
            const scalar_t d_pos = a - p;
            const scalar_t d_neg = a - n;
            
            dist_pos += d_pos * d_pos;
            dist_neg += d_neg * d_neg;
        }
    }
    
    // Handle remaining elements
    #pragma unroll
    for (int i = vec_feat_size * vec_size + lane_id; i < feat_size; i += 32) {
        const int idx = base_idx + i;
        const scalar_t a = anchor[idx];
        const scalar_t p = positive[idx];
        const scalar_t n = negative[idx];
        
        const scalar_t d_pos = a - p;
        const scalar_t d_neg = a - n;
        
        dist_pos += d_pos * d_pos;
        dist_neg += d_neg * d_neg;
    }
    
    // Warp reduction
    dist_pos = warp_reduce_sum(dist_pos);
    dist_neg = warp_reduce_sum(dist_neg);
    
    if (lane_id == 0) {
        const scalar_t loss = max(scalar_t(0.0), sqrt(dist_pos) - sqrt(dist_neg) + margin);
        output[sample_idx] = loss;
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
    
    const int threads_per_block = 128;
    const int warps_per_block = threads_per_block / 32;
    const int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;
    
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