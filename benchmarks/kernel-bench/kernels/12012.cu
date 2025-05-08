#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This version aims to further minimize atomic operations by reorganizing computation to local variable reduction
// whenever possible, and avoiding concurrent memory writes. It uses atomic operations only when writing final
// output to the global output array.

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
        
        // Perform in-block reduction using shared memory
        extern __shared__ scalar_t shmem[];
        scalar_t* shmem_pos = shmem;
        scalar_t* shmem_neg = shmem + blockDim.x;
        
        shmem_pos[threadIdx.x] = dist_pos;
        shmem_neg[threadIdx.x] = dist_neg;
        __syncthreads();
        
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shmem_pos[threadIdx.x] += shmem_pos[threadIdx.x + stride];
                shmem_neg[threadIdx.x] += shmem_neg[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            // Final loss computation for this batch element
            const scalar_t loss = max(scalar_t(0.0), sqrt(shmem_pos[0]) - sqrt(shmem_neg[0]) + margin);
            atomicAdd(&output[batch_idx], loss);
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
        int shared_mem_size = 2 * threads * sizeof(scalar_t);
        triplet_margin_loss_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
