#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Modular device function for warp-level reduction
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    // Reduce within warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Modular device function to accumulate squared distances for a single sample
// Each thread in a warp computes partial sum over feature dimensions
template <typename scalar_t>
__device__ void compute_sample_distance(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    const int sample_idx,
    const int feat_size,
    const int lane_id,
    scalar_t &sum_pos,
    scalar_t &sum_neg
) {
    const int base_idx = sample_idx * feat_size;
    sum_pos = 0;
    sum_neg = 0;
    // Each warp thread processes a subset of features with a stride of warp size (32)
    for (int f = lane_id; f < feat_size; f += 32) {
        const int idx = base_idx + f;
        scalar_t a = anchor[idx];
        scalar_t p = positive[idx];
        scalar_t n = negative[idx];
        scalar_t d_pos = a - p;
        scalar_t d_neg = a - n;
        sum_pos += d_pos * d_pos;
        sum_neg += d_neg * d_neg;
    }
}

// Main kernel: each warp processes one sample from the batch
template <typename scalar_t>
__global__ void triplet_margin_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size
) {
    // Compute global thread id and determine warp id and lane id
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = global_tid / 32;
    const int lane_id = threadIdx.x % 32;

    // Each warp is assigned one sample
    if (warp_id >= batch_size) return;

    scalar_t local_sum_pos = 0;
    scalar_t local_sum_neg = 0;

    // Modular function to accumulate distances
    compute_sample_distance<scalar_t>(anchor, positive, negative, warp_id, feat_size, lane_id, local_sum_pos, local_sum_neg);

    // Perform warp-level reduction
    local_sum_pos = warp_reduce_sum<scalar_t>(local_sum_pos);
    local_sum_neg = warp_reduce_sum<scalar_t>(local_sum_neg);

    // First lane in the warp writes the final result
    if (lane_id == 0) {
        scalar_t loss = sqrt(local_sum_pos) - sqrt(local_sum_neg) + margin;
        output[warp_id] = loss > scalar_t(0) ? loss : scalar_t(0);
    }
}

// CUDA entry point
torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
) {
    TORCH_CHECK(anchor.is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.is_cuda(), "negative must be a CUDA tensor");

    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);

    auto output = torch::zeros({batch_size}, anchor.options());

    // Each sample is handled by one warp (32 threads)
    const int total_threads = batch_size * 32;
    const int threads_per_block = 128;
    const int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel", ([&] {
        triplet_margin_loss_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            margin,
            batch_size,
            feat_size
        );
    }));

    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cuda, "Triplet margin loss forward (CUDA)");
}
