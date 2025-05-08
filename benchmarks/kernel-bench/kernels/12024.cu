#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel assigns one warp per sample to compute the triplet margin loss.
// It minimizes warp divergence by having uniform control flow within each warp and using warp shuffle for reduction.

template <typename scalar_t>
__global__ void triplet_margin_loss_warp_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {

    const int warpSize = 32;
    int warpId = threadIdx.x / warpSize;        // local warp index within block
    int lane = threadIdx.x % warpSize;            // lane index within warp
    int global_warp_id = blockIdx.x * (blockDim.x / warpSize) + warpId; // Each warp handles one sample

    // Only process valid samples; since the condition is uniform across the warp, divergence is minimized
    if (global_warp_id < batch_size) {
        int base = global_warp_id * feat_size;
        scalar_t sum_pos = static_cast<scalar_t>(0);
        scalar_t sum_neg = static_cast<scalar_t>(0);

        // Each thread in the warp processes a strided subset of features
        for (int i = lane; i < feat_size; i += warpSize) {
            scalar_t a = anchor[base + i];
            scalar_t p = positive[base + i];
            scalar_t n = negative[base + i];
            scalar_t d_pos = a - p;
            scalar_t d_neg = a - n;
            sum_pos += d_pos * d_pos;
            sum_neg += d_neg * d_neg;
        }

        // Warp-level reduction using __shfl_down_sync; all threads within a warp follow the same control flow
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_pos += __shfl_down_sync(0xffffffff, sum_pos, offset);
            sum_neg += __shfl_down_sync(0xffffffff, sum_neg, offset);
        }

        // Lane 0 computes the final loss in a branchless manner and writes it to global memory
        if (lane == 0) {
            scalar_t pos_norm = sqrt(sum_pos);
            scalar_t neg_norm = sqrt(sum_neg);
            scalar_t loss = pos_norm - neg_norm + margin;
            // Use branchless max: all lanes would compute the same result, but only lane 0 writes
            loss = loss > static_cast<scalar_t>(0) ? loss : static_cast<scalar_t>(0);
            output[global_warp_id] = loss;
        }
    }
}

// Host function: Launch one warp per sample to compute the loss
// Each warp computes the squared L2 norms across feature dimensions and then applies the Triplet Margin Loss formula.

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

    const int warpSize = 32;
    // Use one warp per sample. Choose threads per block as a multiple of warpSize (e.g., 256 threads).
    const int threads_per_block = 512;
    int warps_per_block = threads_per_block / warpSize;
    int nBlocks = (batch_size + warps_per_block - 1) / warps_per_block;

    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_warp_kernel", ([&] {
        triplet_margin_loss_warp_kernel<scalar_t><<<nBlocks, threads_per_block>>>(
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
    m.def("forward", &triplet_margin_loss_cuda, "Triplet margin loss warp-level optimized forward (CUDA)");
}
