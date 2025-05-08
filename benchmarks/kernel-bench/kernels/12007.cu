#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: Partial reduction kernel
// Each block processes a subset of the feature dimension for one sample.
// The grid is organized as (batch_size * blocksPerSample) blocks.
// For a given block, sample index = blockIdx.x / blocksPerSample, 
// block offset = blockIdx.x % blocksPerSample.

template <typename scalar_t>
__global__ void partial_triplet_margin_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ partial,  // shape: [batch_size * blocksPerSample, 2]
    const int feat_size,
    const int blocksPerSample) {

    // Determine sample and block offset within that sample
    int sample = blockIdx.x / blocksPerSample;
    int block_offset = blockIdx.x % blocksPerSample;
    int base = sample * feat_size;

    scalar_t thread_sum_pos = static_cast<scalar_t>(0);
    scalar_t thread_sum_neg = static_cast<scalar_t>(0);

    // Each block distributes work based on block_offset across threads.
    // Every thread computes indices: f = block_offset + threadIdx.x * blocksPerSample, then f += blocksPerSample * blockDim.x
    for (int f = block_offset + threadIdx.x * blocksPerSample; f < feat_size; f += blocksPerSample * blockDim.x) {
        scalar_t a = anchor[base + f];
        scalar_t p = positive[base + f];
        scalar_t n = negative[base + f];
        scalar_t diff_pos = a - p;
        scalar_t diff_neg = a - n;
        thread_sum_pos += diff_pos * diff_pos;
        thread_sum_neg += diff_neg * diff_neg;
    }

    // Use shared memory for block-level reduction
    extern __shared__ char smem[];
    scalar_t* sdata_pos = reinterpret_cast<scalar_t*>(smem);
    scalar_t* sdata_neg = sdata_pos + blockDim.x;

    sdata_pos[threadIdx.x] = thread_sum_pos;
    sdata_neg[threadIdx.x] = thread_sum_neg;
    __syncthreads();

    // Intra-block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata_pos[threadIdx.x] += sdata_pos[threadIdx.x + s];
            sdata_neg[threadIdx.x] += sdata_neg[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write the partial results to global memory
    if (threadIdx.x == 0) {
        int partial_index = sample * blocksPerSample + block_offset;
        partial[partial_index * 2] = sdata_pos[0];      // sum_pos
        partial[partial_index * 2 + 1] = sdata_neg[0];    // sum_neg
    }
}

// Kernel 2: Final reduction kernel
// Each block processes one sample by summing its partial results and computing the final loss.

template <typename scalar_t>
__global__ void final_triplet_margin_loss_kernel(
    const scalar_t* __restrict__ partial, // shape: [batch_size * blocksPerSample, 2]
    scalar_t* __restrict__ output,        // shape: [batch_size]
    const float margin,
    const int blocksPerSample,
    const int batch_size) {

    int sample = blockIdx.x;  // one block per sample
    scalar_t sum_pos = static_cast<scalar_t>(0);
    scalar_t sum_neg = static_cast<scalar_t>(0);

    for (int i = 0; i < blocksPerSample; i++) {
        int idx = sample * blocksPerSample + i;
        sum_pos += partial[idx * 2];
        sum_neg += partial[idx * 2 + 1];
    }

    scalar_t loss = max(static_cast<scalar_t>(0), sqrt(sum_pos) - sqrt(sum_neg) + static_cast<scalar_t>(margin));
    output[sample] = loss;
}

// Host function: balanced_triplet_margin_loss_cuda
// This function launches two kernels. The first kernel evenly distributes the workload across threads and blocks
// to compute partial reductions. The second kernel aggregates these partial results to compute the final loss
// per sample, and then the mean over all samples is returned.

torch::Tensor balanced_triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {

    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");

    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);

    // Define number of threads per block
    const int threads = 256;
    // Calculate number of blocks per sample so that each thread in a block will get a nearly equal number of features
    int blocksPerSample = (feat_size + threads - 1) / threads;
    // Total blocks for the first kernel
    int totalBlocks = batch_size * blocksPerSample;

    // Allocate partial sum buffer: each block writes two values (sum_pos and sum_neg)
    auto partial = torch::empty({totalBlocks, 2}, anchor.options());
    
    // Launch Kernel 1: partial reduction kernel
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "partial_triplet_margin_loss_kernel", ([&] {
        const int shared_mem_bytes = 2 * threads * sizeof(scalar_t);
        partial_triplet_margin_loss_kernel<scalar_t><<<totalBlocks, threads, shared_mem_bytes>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            partial.data_ptr<scalar_t>(),
            feat_size,
            blocksPerSample);
    }));

    // Allocate output tensor for final loss per sample
    auto output = torch::empty({batch_size}, anchor.options());

    // Launch Kernel 2: final reduction and loss computation kernel
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "final_triplet_margin_loss_kernel", ([&] {
        final_triplet_margin_loss_kernel<scalar_t><<<batch_size, 1>>>(
            partial.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            margin,
            blocksPerSample,
            batch_size);
    }));

    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &balanced_triplet_margin_loss_cuda, "Balanced Triplet Margin Loss forward (CUDA)");
}
