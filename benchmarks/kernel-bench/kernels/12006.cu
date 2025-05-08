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
    
    const int warp_size = 32;
    const unsigned mask = 0xffffffff;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int batch_idx = blockIdx.x;
    
    // Each thread processes multiple elements with striding
    scalar_t dist_pos = 0;
    scalar_t dist_neg = 0;
    
    if (batch_idx < batch_size) {
        const int base_idx = batch_idx * feat_size;
        
        // Stride loop to handle all features
        for (int i = threadIdx.x; i < feat_size; i += blockDim.x) {
            const int idx = base_idx + i;
            const scalar_t a = anchor[idx];
            const scalar_t p = positive[idx];
            const scalar_t n = negative[idx];
            
            const scalar_t d_pos = a - p;
            const scalar_t d_neg = a - n;
            
            dist_pos += d_pos * d_pos;
            dist_neg += d_neg * d_neg;
        }
        
        // Warp-level reduction using shuffle
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset >>= 1) {
            dist_pos += __shfl_down_sync(mask, dist_pos, offset);
            dist_neg += __shfl_down_sync(mask, dist_neg, offset);
        }
        
        // Inter-warp reduction using shared memory
        __shared__ scalar_t shared_pos[32];  // One element per warp
        __shared__ scalar_t shared_neg[32];
        
        if (lane_id == 0) {
            shared_pos[warp_id] = dist_pos;
            shared_neg[warp_id] = dist_neg;
        }
        __syncthreads();
        
        // Final reduction across warps
        if (threadIdx.x < warps_per_block) {
            dist_pos = shared_pos[threadIdx.x];
            dist_neg = shared_neg[threadIdx.x];
            
            #pragma unroll
            for (int offset = warps_per_block/2; offset > 0; offset >>= 1) {
                dist_pos += __shfl_down_sync(mask, dist_pos, offset);
                dist_neg += __shfl_down_sync(mask, dist_neg, offset);
            }
            
            if (threadIdx.x == 0) {
                const scalar_t loss = max(scalar_t(0.0), sqrt(dist_pos) - sqrt(dist_neg) + margin);
                output[batch_idx] = loss;
            }
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
    
    // Use 256 threads per block for optimal occupancy
    const int threads = 256;
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