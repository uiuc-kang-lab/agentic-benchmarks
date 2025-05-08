#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define how many blocks per batch element to partition the feature dimension
#define BLOCKS_PER_BATCH 8

// Kernel 1: Partial reduction kernel
// Each block processes a segment of the feature dimension for a given batch element.
// The grid is configured as: gridDim.x = batch_size * BLOCKS_PER_BATCH
__global__ void triplet_margin_loss_partial(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ partial_sum_pos,  // size: batch_size * BLOCKS_PER_BATCH
    float* __restrict__ partial_sum_neg,  // size: batch_size * BLOCKS_PER_BATCH
    const int feat_size) {

    // Determine which batch element and segment this block will process
    int batch_idx = blockIdx.x / BLOCKS_PER_BATCH;
    int block_offset = blockIdx.x % BLOCKS_PER_BATCH;

    // Divide the feature dimension into BLOCKS_PER_BATCH segments
    int segment = (feat_size + BLOCKS_PER_BATCH - 1) / BLOCKS_PER_BATCH;
    int start = block_offset * segment;
    int end = start + segment;
    if (end > feat_size) end = feat_size;

    int offset = batch_idx * feat_size;
    float sum_pos = 0.0f;
    float sum_neg = 0.0f;

    // Each thread processes a part of the segment
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        float a = __ldg(&anchor[offset + i]);
        float p = __ldg(&positive[offset + i]);
        float n = __ldg(&negative[offset + i]);
        float d_pos = a - p;
        float d_neg = a - n;
        sum_pos += d_pos * d_pos;
        sum_neg += d_neg * d_neg;
    }

    // Use shared memory to reduce sums within the block
    extern __shared__ float sdata[]; // shared memory size: 2 * blockDim.x * sizeof(float)
    float* sdata_pos = sdata;
    float* sdata_neg = sdata + blockDim.x;
    sdata_pos[threadIdx.x] = sum_pos;
    sdata_neg[threadIdx.x] = sum_neg;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata_pos[threadIdx.x] += sdata_pos[threadIdx.x + s];
            sdata_neg[threadIdx.x] += sdata_neg[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write the block's partial sums to global memory
    if (threadIdx.x == 0) {
        int index = batch_idx * BLOCKS_PER_BATCH + block_offset;
        partial_sum_pos[index] = sdata_pos[0];
        partial_sum_neg[index] = sdata_neg[0];
    }
}

// Kernel 2: Final reduction kernel
// Each block processes one batch element by summing the partial results
__global__ void triplet_margin_loss_final(
    const float margin,
    const float* __restrict__ partial_sum_pos,
    const float* __restrict__ partial_sum_neg,
    float* __restrict__ output,  // one result per batch
    const int blocks_per_batch,
    const int batch_size) {

    int batch_idx = blockIdx.x;  
    if (batch_idx >= batch_size) return;

    float total_pos = 0.0f;
    float total_neg = 0.0f;
    int base = batch_idx * blocks_per_batch;

    // Each thread sums a portion of the partial results
    for (int i = threadIdx.x; i < blocks_per_batch; i += blockDim.x) {
        total_pos += partial_sum_pos[base + i];
        total_neg += partial_sum_neg[base + i];
    }

    // Use shared memory to perform final reduction across threads in this block
    extern __shared__ float sdata[]; // shared memory: 2 * blockDim.x * sizeof(float)
    float* sdata_pos = sdata;
    float* sdata_neg = sdata + blockDim.x;
    sdata_pos[threadIdx.x] = total_pos;
    sdata_neg[threadIdx.x] = total_neg;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata_pos[threadIdx.x] += sdata_pos[threadIdx.x + s];
            sdata_neg[threadIdx.x] += sdata_neg[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float loss = sqrtf(sdata_pos[0]) - sqrtf(sdata_neg[0]) + margin;
        output[batch_idx] = (loss > 0.0f) ? loss : 0.0f;
    }
}

// Host function that launches the two kernels
// This implementation partitions the feature dimension of each batch element across multiple blocks,
// ensuring an even distribution of workload across threads and blocks.

torch::Tensor triplet_margin_loss_cuda_balanced(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {

    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");

    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);

    // Allocate temporary arrays for partial sums; one entry per (batch, block segment)
    auto options = anchor.options();
    auto partial_sum_pos = torch::zeros({batch_size * BLOCKS_PER_BATCH}, options);
    auto partial_sum_neg = torch::zeros({batch_size * BLOCKS_PER_BATCH}, options);
    auto output = torch::empty({batch_size}, options);

    // Launch Kernel 1: each block processes a segment of the feature dimension
    int threads = 256;
    int grid1 = batch_size * BLOCKS_PER_BATCH;
    size_t shared_mem_size = 2 * threads * sizeof(float);
    triplet_margin_loss_partial<<<grid1, threads, shared_mem_size>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        partial_sum_pos.data_ptr<float>(),
        partial_sum_neg.data_ptr<float>(),
        feat_size);

    // Launch Kernel 2: reduce partial sums for each batch element
    int threads2 = 32;  // BLOCKS_PER_BATCH is small; 32 threads suffice
    int grid2 = batch_size;
    size_t shared_mem_size2 = 2 * threads2 * sizeof(float);
    triplet_margin_loss_final<<<grid2, threads2, shared_mem_size2>>>(
        margin,
        partial_sum_pos.data_ptr<float>(),
        partial_sum_neg.data_ptr<float>(),
        output.data_ptr<float>(),
        BLOCKS_PER_BATCH,
        batch_size);

    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cuda_balanced, "Triplet margin loss forward balanced optimization (CUDA)");
}
