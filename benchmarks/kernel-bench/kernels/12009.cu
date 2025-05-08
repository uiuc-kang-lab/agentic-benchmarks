#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Store frequently accessed read-only margin value in constant memory
__constant__ float c_margin;

// Kernel using warp-level primitives and constant memory.
// Each block processes one batch sample and reduces the feature dimension using warp shuffles.

template <typename scalar_t>
__global__ void const_mem_warp_triplet_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const int feat_size) {

    // Each block corresponds to one batch sample
    const int batch_idx = blockIdx.x;
    const int base_idx = batch_idx * feat_size;

    // Each thread accumulates partial sum for its assigned features
    scalar_t sum_pos = static_cast<scalar_t>(0);
    scalar_t sum_neg = static_cast<scalar_t>(0);

    for (int i = threadIdx.x; i < feat_size; i += blockDim.x) {
        const int idx = base_idx + i;
        const scalar_t a = anchor[idx];
        const scalar_t p = positive[idx];
        const scalar_t n = negative[idx];
        scalar_t diff_pos = a - p;
        scalar_t diff_neg = a - n;
        sum_pos += diff_pos * diff_pos;
        sum_neg += diff_neg * diff_neg;
    }

    // Use warp-level reduction using shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_pos += __shfl_down_sync(mask, sum_pos, offset);
        sum_neg += __shfl_down_sync(mask, sum_neg, offset);
    }

    // Each warp's lane 0 holds the partial sum for that warp
    // Use shared memory to further reduce across warps
    __shared__ scalar_t shared_pos[32];
    __shared__ scalar_t shared_neg[32];

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_pos[warp_id] = sum_pos;
        shared_neg[warp_id] = sum_neg;
    }
    __syncthreads();

    // First warp reduces the partial sums from each warp
    scalar_t total_pos = static_cast<scalar_t>(0);
    scalar_t total_neg = static_cast<scalar_t>(0);
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warp_id == 0) {
        if (lane < num_warps) {
            total_pos = shared_pos[lane];
            total_neg = shared_neg[lane];
        } else {
            total_pos = static_cast<scalar_t>(0);
            total_neg = static_cast<scalar_t>(0);
        }
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            total_pos += __shfl_down_sync(mask, total_pos, offset);
            total_neg += __shfl_down_sync(mask, total_neg, offset);
        }
        if (lane == 0) {
            // Final loss computation using margin from constant memory
            scalar_t loss = sqrt(total_pos) - sqrt(total_neg) + c_margin;
            loss = loss < static_cast<scalar_t>(0) ? static_cast<scalar_t>(0) : loss;
            output[batch_idx] = loss;
        }
    }
}

// Host function that sets up and launches the kernel

torch::Tensor const_mem_warp_triplet_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {

    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");

    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);

    // Allocate output tensor (one loss value per sample)
    auto output = torch::zeros({batch_size}, anchor.options());

    // Copy margin value to constant memory
    cudaMemcpyToSymbolAsync(c_margin, &margin, sizeof(float), 0, cudaMemcpyHostToDevice);

    // Launch configuration: one block per sample
    const int threads = 256;
    dim3 blocks(batch_size);

    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "const_mem_warp_triplet_kernel", ([&] {
        const_mem_warp_triplet_kernel<scalar_t><<<blocks, threads>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            feat_size);
    }));

    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &const_mem_warp_triplet_cuda, "Triplet margin loss forward with constant memory and warp-level reduction (CUDA)");
}
