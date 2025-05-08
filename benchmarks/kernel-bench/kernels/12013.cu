#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel with coalesced memory access patterns

template <typename scalar_t>
__global__ void triplet_margin_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_size = 32;
    const unsigned mask = 0xffffffff;

    const int batch_idx = (tid / warp_size) % batch_size;
    const int lane_idx = tid % warp_size;
    const int feat_offset = (tid / (warp_size * batch_size)) * warp_size;

    if (batch_idx < batch_size && feat_offset < feat_size) {
        scalar_t dist_pos = 0;
        scalar_t dist_neg = 0;
        if (feat_offset + lane_idx < feat_size) {
            const int idx = batch_idx * feat_size + feat_offset + lane_idx;
            const scalar_t a = anchor[idx];
            const scalar_t p = positive[idx];
            const scalar_t n = negative[idx];

            // Compute distance components
            const scalar_t d_pos = a - p;
            const scalar_t d_neg = a - n;

            // Squared distances
            dist_pos = d_pos * d_pos;
            dist_neg = d_neg * d_neg;
        }

        // Warp-level reduction for features that fit in a warp
        #pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
            dist_pos += __shfl_down_sync(mask, dist_pos, offset);
            dist_neg += __shfl_down_sync(mask, dist_neg, offset);
        }

        if (lane_idx == 0) {
            atomicAdd(&output[batch_idx], max(scalar_t(0.0), sqrt(dist_pos) - sqrt(dist_neg) + margin));
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
    const int blocks = (batch_size * feat_size + threads * 32 - 1) / (threads * 32);

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
