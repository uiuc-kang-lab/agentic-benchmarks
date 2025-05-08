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
__global__ void optimized_triplet_margin_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    extern __shared__ scalar_t shared_mem[];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = tid / feat_size;
    const int feat_idx = tid % feat_size;
    const int lane_id = threadIdx.x % 32;

    scalar_t dist_pos = 0.0;
    scalar_t dist_neg = 0.0;

    if (batch_idx < batch_size) {
        if (feat_idx < feat_size) {
            const int idx = batch_idx * feat_size + feat_idx;
            const scalar_t a = anchor[idx];
            const scalar_t p = positive[idx];
            const scalar_t n = negative[idx];

            const scalar_t d_pos = a - p;
            const scalar_t d_neg = a - n;

            dist_pos = d_pos * d_pos;
            dist_neg = d_neg * d_neg;
        }

        dist_pos = warp_reduce_sum(dist_pos);
        dist_neg = warp_reduce_sum(dist_neg);

        // Use shared memory to store partial results from each warp
        if (lane_id == 0) {
            shared_mem[threadIdx.x / 32] = dist_pos;
            shared_mem[threadIdx.x / 32 + blockDim.x / 32] = dist_neg;
        }

        __syncthreads();

        // Reduce within block
        if (lane_id == 0) {
            scalar_t block_dist_pos = 0.0;
            scalar_t block_dist_neg = 0.0;
            for (int i = 0; i < blockDim.x / 32; ++i) {
                block_dist_pos += shared_mem[i];
                block_dist_neg += shared_mem[i + blockDim.x / 32];
            }

            if (threadIdx.x == 0) {
                // Final loss computation for this batch element
                const scalar_t loss = max(scalar_t(0.0), sqrt(block_dist_pos) - sqrt(block_dist_neg) + margin);
                output[batch_idx] = loss;
            }
        }
    }
}

torch::Tensor optimized_triplet_margin_loss_cuda(
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
    const int blocks = (batch_size * feat_size + threads_per_block - 1) / threads_per_block;
    const int shared_memory_size = 2 * (threads_per_block / 32) * sizeof(scalar_t);
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "optimized_triplet_margin_loss_kernel", ([&] {
        optimized_triplet_margin_loss_kernel<scalar_t><<<blocks, threads_per_block, shared_memory_size>>>(
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
    m.def("forward", &optimized_triplet_margin_loss_cuda, "Optimized Triplet margin loss forward (CUDA)");
}