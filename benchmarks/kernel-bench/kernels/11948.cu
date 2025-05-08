#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel using balanced workload distribution and vectorized loads

template <typename scalar_t>
__global__ void balanced_triplet_margin_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    scalar_t local_dist_pos = 0;
    scalar_t local_dist_neg = 0;

    // Distribute workload evenly across all threads
    for (int idx = tid; idx < batch_size * feat_size; idx += total_threads) {
        int batch_idx = idx / feat_size;
        int feat_idx = idx % feat_size;

        scalar_t a = __ldg(&anchor[batch_idx * feat_size + feat_idx]);
        scalar_t p = __ldg(&positive[batch_idx * feat_size + feat_idx]);
        scalar_t n = __ldg(&negative[batch_idx * feat_size + feat_idx]);

        scalar_t d_pos = a - p;
        scalar_t d_neg = a - n;

        local_dist_pos += d_pos * d_pos;
        local_dist_neg += d_neg * d_neg;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_dist_pos += __shfl_down_sync(0xffffffff, local_dist_pos, offset);
        local_dist_neg += __shfl_down_sync(0xffffffff, local_dist_neg, offset);
    }

    __shared__ scalar_t shared_pos[32];
    __shared__ scalar_t shared_neg[32];

    int lane = tid % 32;
    int warp_id = tid / 32;

    if (lane == 0) {
        shared_pos[warp_id] = local_dist_pos;
        shared_neg[warp_id] = local_dist_neg;
    }
    __syncthreads();

    if (warp_id == 0) {
        scalar_t block_sum_pos = (tid < (blockDim.x / 32)) ? shared_pos[lane] : 0;
        scalar_t block_sum_neg = (tid < (blockDim.x / 32)) ? shared_neg[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum_pos += __shfl_down_sync(0xffffffff, block_sum_pos, offset);
            block_sum_neg += __shfl_down_sync(0xffffffff, block_sum_neg, offset);
        }
        if (lane == 0) {
            scalar_t loss = sqrt(block_sum_pos) - sqrt(block_sum_neg) + margin;
            atomicAdd(&output[warp_id], loss < scalar_t(0) ? scalar_t(0) : loss);
        }
    }
}

// Host function to launch the kernel
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
    
    // Launch with enough blocks to cover all data
    const int threads_per_block = 256;
    const int num_blocks = (batch_size * feat_size + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "balanced_triplet_margin_loss_kernel", ([&] {
        balanced_triplet_margin_loss_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
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
