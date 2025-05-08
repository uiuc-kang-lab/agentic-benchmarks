#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void triplet_margin_loss_kernel_warp_optimized(
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
    
    if (batch_idx < batch_size) {
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
        extern __shared__ scalar_t shared_mem[];
        shared_mem[threadIdx.x] = dist_pos - dist_neg;
        __syncthreads();
        
        // Reduce within feature dimension
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            // Final loss computation for this batch element
            const scalar_t loss = max(scalar_t(0.0), sqrt(shared_mem[0]) + margin);
            atomicAdd(&output[batch_idx], loss);
        }
    }
}

torch::Tensor triplet_margin_loss_cuda_warp_optimized(
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
    const int shared_mem_size = threads * sizeof(scalar_t);
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel_warp_optimized", ([&] {
        triplet_margin_loss_kernel_warp_optimized<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &triplet_margin_loss_cuda_warp_optimized, "Triplet margin loss forward (CUDA)");
}