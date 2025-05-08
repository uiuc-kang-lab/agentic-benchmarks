#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ void warp_reduce_sum(scalar_t& val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
}

template <typename scalar_t>
__global__ void triplet_margin_loss_kernel(
    const scalar_t* anchor,
    const scalar_t* positive,
    const scalar_t* negative,
    scalar_t* output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    if (warp_id >= batch_size) return;
    
    scalar_t dist_pos = 0.0;
    scalar_t dist_neg = 0.0;
    
    for (int i = lane_id; i < feat_size; i += 32) {
        const int idx = warp_id * feat_size + i;
        const scalar_t a = anchor[idx];
        const scalar_t p = positive[idx];
        const scalar_t n = negative[idx];
        
        const scalar_t d_pos = a - p;
        const scalar_t d_neg = a - n;
        
        dist_pos += d_pos * d_pos;
        dist_neg += d_neg * d_neg;
    }
    
    warp_reduce_sum(dist_pos);
    warp_reduce_sum(dist_neg);
    
    if (lane_id == 0) {
        const scalar_t loss = fmaxf(0.0, sqrtf(dist_pos) - sqrtf(dist_neg) + static_cast<scalar_t>(margin));
        output[warp_id] = loss;
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
    const int warps_per_block = threads_per_block / 32;
    const int blocks = (batch_size + warps_per_block - 1) / warps_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel", ([&] {
        triplet_margin_loss_kernel<scalar_t><<<blocks, threads_per_block>>>(
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