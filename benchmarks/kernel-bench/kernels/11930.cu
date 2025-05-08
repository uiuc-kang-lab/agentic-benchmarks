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
    
    extern __shared__ scalar_t shared_mem[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    const int batch_idx = bid;
    
    if (batch_idx >= batch_size) return;
    
    scalar_t dist_pos = 0.0;
    scalar_t dist_neg = 0.0;
    
    #pragma unroll 4
    for (int feat_idx = tid; feat_idx < feat_size; feat_idx += blockDim.x) {
        const int idx = batch_idx * feat_size + feat_idx;
        const scalar_t a = anchor[idx];
        const scalar_t p = positive[idx];
        const scalar_t n = negative[idx];
        
        const scalar_t d_pos = a - p;
        const scalar_t d_neg = a - n;
        
        dist_pos += d_pos * d_pos;
        dist_neg += d_neg * d_neg;
    }
    
    dist_pos = warp_reduce_sum(dist_pos);
    dist_neg = warp_reduce_sum(dist_neg);
    
    if (lane_id == 0) {
        shared_mem[warp_id*2] = dist_pos;
        shared_mem[warp_id*2 + 1] = dist_neg;
    }
    
    __syncthreads();
    
    if (warp_id == 0 && lane_id < (blockDim.x/warpSize)) {
        dist_pos = shared_mem[lane_id*2];
        dist_neg = shared_mem[lane_id*2 + 1];
        
        dist_pos = warp_reduce_sum(dist_pos);
        dist_neg = warp_reduce_sum(dist_neg);
        
        if (lane_id == 0) {
            output[batch_idx] = max(scalar_t(0.0), sqrt(dist_pos) - sqrt(dist_neg) + margin);
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
    
    const int block_size = 256;
    const int num_blocks = batch_size;
    const int shared_mem_size = (block_size/32) * 2 * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel", ([&] {
        triplet_margin_loss_kernel<scalar_t><<<num_blocks, block_size, shared_mem_size>>>(
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