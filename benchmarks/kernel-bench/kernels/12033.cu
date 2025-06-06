#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using shared memory and warp-level primitives for efficient reduction

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
    const int batch_idx = tid / feat_size;
    const int feat_idx = tid % feat_size;
    const int warp_size = 32;
    const unsigned mask = 0xffffffff;
    
    if (batch_idx < batch_size && feat_idx < feat_size) {
        const int idx = batch_idx * feat_size + feat_idx;
        const scalar_t a = anchor[idx];
        const scalar_t p = positive[idx];
        const scalar_t n = negative[idx];
        
        // Compute distance components
        const scalar_t d_pos = a - p;
        const scalar_t d_neg = a - n;
        
        // Squared distances
        scalar_t dist_pos = d_pos * d_pos;
        scalar_t dist_neg = d_neg * d_neg;
        
        // Use shared memory for reduction
        __shared__ scalar_t shared_pos[256];
        __shared__ scalar_t shared_neg[256];
        
        shared_pos[threadIdx.x] = dist_pos;
        shared_neg[threadIdx.x] = dist_neg;
        __syncthreads();
        
        // Intra-block reduction using shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_pos[threadIdx.x] += shared_pos[threadIdx.x + stride];
                shared_neg[threadIdx.x] += shared_neg[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        // Final reduction using warp shuffle
        if (threadIdx.x < warp_size) {
            dist_pos = shared_pos[threadIdx.x];
            dist_neg = shared_neg[threadIdx.x];
            
            #pragma unroll
            for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
                dist_pos += __shfl_down_sync(mask, dist_pos, offset);
                dist_neg += __shfl_down_sync(mask, dist_neg, offset);
            }
        }
        
        if (feat_idx == 0 && threadIdx.x == 0) {
            // Final loss computation for this batch element
            const scalar_t loss = max(scalar_t(0.0), dist_pos - dist_neg + margin);
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
    
    const int threads = 256;
    const int blocks = (batch_size * feat_size + threads - 1) / threads;
    
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
