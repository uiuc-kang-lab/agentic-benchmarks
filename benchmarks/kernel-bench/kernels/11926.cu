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
    
    extern __shared__ scalar_t sdata[];
    scalar_t* shared_anchor = sdata;
    scalar_t* shared_positive = shared_anchor + blockDim.x;
    scalar_t* shared_negative = shared_positive + blockDim.x;
    
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    
    const int start_idx = block_idx * blockDim.x + thread_idx;
    const int step_size = blockDim.x * gridDim.x;
    
    scalar_t dist_pos = 0.0;
    scalar_t dist_neg = 0.0;
    
    for (int idx = start_idx; idx < batch_size * feat_size; idx += step_size) {
        const int feat_idx = idx % feat_size;
        const int batch_idx = idx / feat_size;

        if (feat_idx < feat_size) {
            shared_anchor[thread_idx] = anchor[idx];
            shared_positive[thread_idx] = positive[idx];
            shared_negative[thread_idx] = negative[idx];
        }

        __syncthreads();

        const scalar_t a = shared_anchor[feat_idx];
        const scalar_t p = shared_positive[feat_idx];
        const scalar_t n = shared_negative[feat_idx];

        const scalar_t d_pos = a - p;
        const scalar_t d_neg = a - n;

        dist_pos += d_pos * d_pos;
        dist_neg += d_neg * d_neg;

        __syncthreads();
    }
 
    if (feat_idx < feat_size && thread_idx == 0) {
        scalar_t final_dist_pos = sqrt(dist_pos);
        scalar_t final_dist_neg = sqrt(dist_neg);
        scalar_t loss = max(scalar_t(0.0), final_dist_pos - final_dist_neg + margin);
        atomicAdd(&output[block_idx], loss);
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
    const int blocks = (batch_size * feat_size + threads_per_block - 1) / threads_per_block;
    
    const int shared_mem_size = 3 * threads_per_block * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel", ([&] {
        triplet_margin_loss_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
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
