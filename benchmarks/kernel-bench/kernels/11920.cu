#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void triplet_margin_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    __shared__ scalar_t shared_mem[32];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int sample_idx = bid;
    
    if (sample_idx >= batch_size) return;
    
    // Process features in a coalesced manner
    scalar_t dist_pos = 0.0f;
    scalar_t dist_neg = 0.0f;
    
    const int base_idx = sample_idx * feat_size;
    
    // Stride through features with multiple threads for coalesced access
    for (int feat_idx = tid; feat_idx < feat_size; feat_idx += blockDim.x) {
        const int idx = base_idx + feat_idx;
        const scalar_t a = anchor[idx];
        const scalar_t p = positive[idx];
        const scalar_t n = negative[idx];
        
        const scalar_t d_pos = a - p;
        const scalar_t d_neg = a - n;
        
        dist_pos += d_pos * d_pos;
        dist_neg += d_neg * d_neg;
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dist_pos += __shfl_down_sync(0xffffffff, dist_pos, offset);
        dist_neg += __shfl_down_sync(0xffffffff, dist_neg, offset);
    }
    
    // First thread in warp writes result
    if (tid == 0) {
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
    
    const int threads = 128;
    const int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel", ([&] {
        triplet_margin_loss_kernel<scalar_t><<<blocks, threads>>>(
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