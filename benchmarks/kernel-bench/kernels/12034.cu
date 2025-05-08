#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel processes one sample per block, and leverages shared memory to cache tiles of the feature vectors
// for anchor, positive, and negative in order to reduce global memory latency.

template <typename scalar_t>
__global__ void shared_mem_triplet_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int feat_size) {
    
    // Each block processes one sample
    const int sample = blockIdx.x;
    const int base_idx = sample * feat_size;
    
    // Declare shared memory for caching a tile of the feature vectors.
    // We allocate three arrays: one each for anchor, positive, and negative data.
    extern __shared__ char smem[];
    scalar_t* s_anchor = reinterpret_cast<scalar_t*>(smem);
    scalar_t* s_positive = s_anchor + blockDim.x;
    scalar_t* s_negative = s_positive + blockDim.x;
    
    // Each thread accumulates its partial sums for the squared differences
    scalar_t local_sum_pos = static_cast<scalar_t>(0);
    scalar_t local_sum_neg = static_cast<scalar_t>(0);
    
    // Process the feature vector in tiles of size equal to the blockDim
    for (int tile = 0; tile < feat_size; tile += blockDim.x) {
        int index = tile + threadIdx.x;

        // Load data into shared memory if index is within bounds; otherwise load 0
        if (index < feat_size) {
            s_anchor[threadIdx.x] = anchor[base_idx + index];
            s_positive[threadIdx.x] = positive[base_idx + index];
            s_negative[threadIdx.x] = negative[base_idx + index];
        } else {
            s_anchor[threadIdx.x] = static_cast<scalar_t>(0);
            s_positive[threadIdx.x] = static_cast<scalar_t>(0);
            s_negative[threadIdx.x] = static_cast<scalar_t>(0);
        }
        __syncthreads();
        
        // Each thread processes its loaded element from shared memory
        if (index < feat_size) {
            scalar_t a = s_anchor[threadIdx.x];
            scalar_t p = s_positive[threadIdx.x];
            scalar_t n = s_negative[threadIdx.x];
            scalar_t diff_pos = a - p;
            scalar_t diff_neg = a - n;
            local_sum_pos += diff_pos * diff_pos;
            local_sum_neg += diff_neg * diff_neg;
        }
        __syncthreads(); // Ensure all threads are done processing this tile before loading the next
    }
    
    // Reduce the partial sums across the block using shared memory reduction
    // Reuse s_anchor for reduction of positive sums and s_positive for negative sums
    s_anchor[threadIdx.x] = local_sum_pos;
    s_positive[threadIdx.x] = local_sum_neg;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_anchor[threadIdx.x] += s_anchor[threadIdx.x + stride];
            s_positive[threadIdx.x] += s_positive[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the final loss for this sample
    if (threadIdx.x == 0) {
        scalar_t total_pos = s_anchor[0];
        scalar_t total_neg = s_positive[0];
        scalar_t loss = max(static_cast<scalar_t>(0), sqrt(total_pos) - sqrt(total_neg) + static_cast<scalar_t>(margin));
        output[sample] = loss;
    }
}

// Host function: Launch one block per sample and leverage shared memory tiling

torch::Tensor shared_mem_triplet_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {

    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");

    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);

    // Output: one loss per sample
    auto output = torch::empty({batch_size}, anchor.options());

    // Launch one block per sample
    int threads = 256;
    if (feat_size < threads) {
        threads = feat_size;
    }
    dim3 blocks(batch_size);
    dim3 threadsPerBlock(threads);

    // Shared memory: 3 arrays of size 'threads' of type scalar_t
    size_t shared_mem_size = 3 * threads * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "shared_mem_triplet_loss_kernel", ([&] {
        shared_mem_triplet_loss_kernel<scalar_t><<<blocks, threadsPerBlock, shared_mem_size>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            margin,
            feat_size);
    }));

    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_mem_triplet_loss_cuda, "Shared Memory Triplet Margin Loss forward (CUDA)");
}
