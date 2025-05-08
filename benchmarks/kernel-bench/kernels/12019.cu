#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ float d_margin;

template <typename scalar_t>
__global__ void triplet_margin_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int feat_size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_warps = blockDim.x / warp_size;
    const int batch_idx = tid / feat_size;
    const int feat_idx = tid % feat_size;
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const unsigned mask = 0xffffffff;
    
    __shared__ scalar_t shared_pos[4];
    __shared__ scalar_t shared_neg[4];
    
    scalar_t dist_pos = 0;
    scalar_t dist_neg = 0;
    
    if (batch_idx < batch_size) {
        #pragma unroll 4
        for (int f = feat_idx; f < feat_size; f += blockDim.x) {
            const int idx = batch_idx * feat_size + f;
            const scalar_t a = anchor[idx];
            const scalar_t p = positive[idx];
            const scalar_t n = negative[idx];
            
            const scalar_t d_pos = a - p;
            const scalar_t d_neg = a - n;
            
            dist_pos += d_pos * d_pos;
            dist_neg += d_neg * d_neg;
        }
        
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset >>= 1) {
            dist_pos += __shfl_down_sync(mask, dist_pos, offset);
            dist_neg += __shfl_down_sync(mask, dist_neg, offset);
        }
        
        if (lane_id == 0) {
            shared_pos[warp_id] = dist_pos;
            shared_neg[warp_id] = dist_neg;
        }
        __syncthreads();
        
        if (warp_id == 0 && lane_id < 4) {
            dist_pos = shared_pos[lane_id];
            dist_neg = shared_neg[lane_id];
            
            #pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                dist_pos += __shfl_down_sync(mask, dist_pos, offset);
                dist_neg += __shfl_down_sync(mask, dist_neg, offset);
            }
            
            if (lane_id == 0) {
                const scalar_t loss = max(scalar_t(0.0), sqrt(dist_pos) - sqrt(dist_neg) + d_margin);
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
    
    cudaMemcpyToSymbol(d_margin, &margin, sizeof(float));
    
    auto output = torch::zeros({batch_size}, anchor.options());
    
    const int threads = 128;
    const int blocks = (batch_size * feat_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel", ([&] {
        triplet_margin_loss_kernel<scalar_t><<<blocks, threads>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            feat_size);
    }));
    
    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cuda, "Triplet margin loss forward (CUDA)");
}