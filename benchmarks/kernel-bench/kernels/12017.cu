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
    const int feat_size) {

    // Each block computes the loss for one batch sample. blockIdx.x indexes the batch.
    const int batch_idx = blockIdx.x;

    // Each thread computes a partial sum over the feature dimensions
    scalar_t sum_pos = 0;
    scalar_t sum_neg = 0;
    for (int i = threadIdx.x; i < feat_size; i += blockDim.x) {
        int idx = batch_idx * feat_size + i;
        scalar_t a = anchor[idx];
        scalar_t p = positive[idx];
        scalar_t n = negative[idx];
        scalar_t d_pos = a - p;
        scalar_t d_neg = a - n;
        sum_pos += d_pos * d_pos;
        sum_neg += d_neg * d_neg;
    }

    // Reduction within the block using shared memory
    extern __shared__ char smem[];
    scalar_t* sdata_pos = reinterpret_cast<scalar_t*>(smem);
    scalar_t* sdata_neg = sdata_pos + blockDim.x;

    sdata_pos[threadIdx.x] = sum_pos;
    sdata_neg[threadIdx.x] = sum_neg;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata_pos[threadIdx.x] += sdata_pos[threadIdx.x + s];
            sdata_neg[threadIdx.x] += sdata_neg[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 computes the final loss for this batch sample
    if (threadIdx.x == 0) {
        scalar_t pos_dist = sqrt(sdata_pos[0]);
        scalar_t neg_dist = sqrt(sdata_neg[0]);
        scalar_t loss = max(scalar_t(0.0), pos_dist - neg_dist + margin);
        output[batch_idx] = loss;
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
