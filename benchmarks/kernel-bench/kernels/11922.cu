#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void triplet_margin_loss_kernel(
    const scalar_t* anchor,
    const scalar_t* positive,
    const scalar_t* negative,
    scalar_t* output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    // 2D thread organization: x for features, y for batch samples
    const int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Shared memory for partial results
    extern __shared__ char shared_mem[];
    scalar_t* shared_pos = (scalar_t*)shared_mem;
    scalar_t* shared_neg = shared_pos + blockDim.y;
    
    const int shared_idx = threadIdx.y;
    shared_pos[shared_idx] = 0;
    shared_neg[shared_idx] = 0;
    
    if (batch_idx < batch_size) {
        scalar_t dist_pos = 0;
        scalar_t dist_neg = 0;
        
        // Each thread processes multiple features if needed
        for (int f = feat_idx; f < feat_size; f += blockDim.x * gridDim.x) {
            const int idx = batch_idx * feat_size + f;
            const scalar_t a = anchor[idx];
            const scalar_t p = positive[idx];
            const scalar_t n = negative[idx];
            
            const scalar_t d_pos = a - p;
            const scalar_t d_neg = a - n;
            
            dist_pos += d_pos * d_pos;
            dist_neg += d_neg * d_neg;
        }
        
        // Add partial results to shared memory
        atomicAdd(&shared_pos[shared_idx], dist_pos);
        atomicAdd(&shared_neg[shared_idx], dist_neg);
    }
    
    __syncthreads();
    
    // Only one thread per batch sample computes final result
    if (threadIdx.x == 0 && batch_idx < batch_size) {
        const scalar_t final_dist_pos = sqrt(shared_pos[shared_idx]);
        const scalar_t final_dist_neg = sqrt(shared_neg[shared_idx]);
        output[batch_idx] = max(scalar_t(0.0), final_dist_pos - final_dist_neg + margin);
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
    
    // 2D grid configuration
    dim3 threads_per_block(16, 16);
    dim3 num_blocks(
        (feat_size + threads_per_block.x - 1) / threads_per_block.x,
        (batch_size + threads_per_block.y - 1) / threads_per_block.y
    );
    
    auto output = torch::zeros({batch_size}, anchor.options());
    
    const int shared_mem_size = 2 * threads_per_block.y * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel", ([&] {
        triplet_margin_loss_kernel<scalar_t><<<num_blocks, threads_per_block, shared_mem_size>>>(
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