#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: Tiled reduction across the feature dimension
// Grid dimensions: gridDim.x = batch_size, gridDim.y = num_tiles (tiles covering the feature dimension)
// Each block processes a tile of features for one batch element and reduces its partial sums in shared memory
// Then, the block writes its partial sums using atomicAdd to global accumulators for that batch element

__global__ void tiled_triplet_loss_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ pos_partial, // global accumulator for positive distance sums per batch element
    float* __restrict__ neg_partial, // global accumulator for negative distance sums per batch element
    const int feat_size) {

    // Each block is assigned a specific batch element and a tile of the feature dimension
    int batch_idx = blockIdx.x;
    int tile_start = blockIdx.y * blockDim.x;  // starting index for this tile in the feature dimension
    int tid = threadIdx.x;
    
    float pos_sum = 0.f;
    float neg_sum = 0.f;

    // Loop over the tile with a stride covering the entire feature dimension for this batch element
    // The stride is blockDim.x * gridDim.y, so that if the tile does not cover all features, each thread will
    // process additional elements in the same batch element
    for (int i = tile_start + tid; i < feat_size; i += blockDim.x * gridDim.y) {
        int idx = batch_idx * feat_size + i;
        float a = __ldg(anchor + idx);
        float p = __ldg(positive + idx);
        float n = __ldg(negative + idx);
        float diff = a - p;
        pos_sum += diff * diff;
        diff = a - n;
        neg_sum += diff * diff;
    }

    // Intra-block reduction using shared memory
    extern __shared__ float sdata[]; // shared memory: first blockDim.x floats for pos, next blockDim.x for neg
    float* shared_pos = sdata;
    float* shared_neg = sdata + blockDim.x;
    shared_pos[tid] = pos_sum;
    shared_neg[tid] = neg_sum;
    __syncthreads();

    // Reduce the block's values
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_pos[tid] += shared_pos[tid + s];
            shared_neg[tid] += shared_neg[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 in this block adds the block's partial sum to the global accumulators
    if (tid == 0) {
        atomicAdd(&pos_partial[batch_idx], shared_pos[0]);
        atomicAdd(&neg_partial[batch_idx], shared_neg[0]);
    }
}

// Kernel 2: Final loss computation
// For each batch element, compute loss = max(sqrt(pos_sum) - sqrt(neg_sum) + margin, 0)
__global__ void final_loss_kernel(
    const float* __restrict__ pos_partial,
    const float* __restrict__ neg_partial,
    float* __restrict__ loss,
    const float margin,
    const int batch_size) {

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size) {
        float pos_sum = pos_partial[batch_idx];
        float neg_sum = neg_partial[batch_idx];
        float l = sqrtf(pos_sum) - sqrtf(neg_sum) + margin;
        loss[batch_idx] = (l > 0.f) ? l : 0.f;
    }
}


// Host launcher
// This function allocates global accumulators for each batch element and launches two kernels:
// 1. tiled_triplet_loss_kernel distributes the feature dimension across blocks for each batch element
//    and accumulates partial reduction results via atomicAdd.
// 2. final_loss_kernel computes the final loss per batch element, and the mean loss is returned.

torch::Tensor triplet_margin_loss_cuda_distributed(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {

    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");

    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);

    // Allocate global accumulators for each batch element (initialized to zero)
    auto pos_partial = torch::zeros({batch_size}, anchor.options());
    auto neg_partial = torch::zeros({batch_size}, anchor.options());
    // Allocate tensor for final loss per batch element
    auto loss_tensor = torch::empty({batch_size}, anchor.options());

    // Kernel 1 configuration
    int threads = 256;  
    // Compute number of tiles needed to cover the feature dimension
    int num_tiles = (feat_size + threads - 1) / threads;
    // Grid: one dimension for batch elements and one for tiles along the feature dimension
    dim3 blocks(batch_size, num_tiles);
    size_t shared_mem_size = 2 * threads * sizeof(float);  // for shared_pos and shared_neg

    tiled_triplet_loss_kernel<<<blocks, threads, shared_mem_size>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        pos_partial.data_ptr<float>(),
        neg_partial.data_ptr<float>(),
        feat_size);

    // Kernel 2 configuration: one thread per batch element for final loss computation
    int threads2 = 256;
    int blocks2 = (batch_size + threads2 - 1) / threads2;

    final_loss_kernel<<<blocks2, threads2>>>(
        pos_partial.data_ptr<float>(),
        neg_partial.data_ptr<float>(),
        loss_tensor.data_ptr<float>(),
        margin,
        batch_size);

    // Return the mean loss across the batch
    return loss_tensor.mean();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cuda_distributed, "Triplet margin loss forward with distributed workload (CUDA)");
}
