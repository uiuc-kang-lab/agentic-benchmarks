#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define tile size for shared memory tiling
#define TILE_SIZE 128

// Kernel that leverages shared memory to cache tiles of predictions and targets
// to reduce global memory latency. Each block processes one sample (row) and
// iterates over the row in tiles. Partial sums are reduced within the block using
// a dedicated shared memory region.
__global__ void cosine_similarity_loss_kernel_shared(const float* __restrict__ predictions,
                                                       const float* __restrict__ targets,
                                                       float* output,
                                                       int N,
                                                       int D) {
    // Each block processes one row (sample)
    int row = blockIdx.x;
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Dynamic shared memory layout:
    // [0, TILE_SIZE)           -> s_pred: tile for predictions
    // [TILE_SIZE, 2*TILE_SIZE)  -> s_target: tile for targets
    // [2*TILE_SIZE, 2*TILE_SIZE + blockSize)              -> r_dot
    // [2*TILE_SIZE + blockSize, 2*TILE_SIZE + 2*blockSize)  -> r_pred_sq
    // [2*TILE_SIZE + 2*blockSize, 2*TILE_SIZE + 3*blockSize) -> r_target_sq
    extern __shared__ float s_data[];
    float* s_pred = s_data;
    float* s_target = s_pred + TILE_SIZE;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Process the row in tiles of TILE_SIZE elements
    for (int offset = 0; offset < D; offset += TILE_SIZE) {
        // Each thread with index less than TILE_SIZE loads one element into shared memory
        int tile_index = threadIdx.x;
        if (tile_index < TILE_SIZE) {
            int col = offset + tile_index;
            if (col < D) {
                s_pred[tile_index] = pred_row[col];
                s_target[tile_index] = target_row[col];
            } else {
                s_pred[tile_index] = 0.0f;
                s_target[tile_index] = 0.0f;
            }
        }
        __syncthreads();

        // Determine the number of valid elements in this tile
        int current_tile_size = ((D - offset) < TILE_SIZE) ? (D - offset) : TILE_SIZE;
        // Each thread processes a subset of elements in the tile
        for (int j = tid; j < current_tile_size; j += blockSize) {
            float p = s_pred[j];
            float t = s_target[j];
            sum_dot += p * t;
            sum_pred_sq += p * p;
            sum_target_sq += t * t;
        }
        __syncthreads();
    }

    // Allocate shared memory for block-level reduction using the remaining dynamic shared memory
    float* r_dot = s_data + 2 * TILE_SIZE;
    float* r_pred_sq = r_dot + blockSize;
    float* r_target_sq = r_pred_sq + blockSize;

    // Each thread writes its partial sums into the reduction arrays
    r_dot[tid] = sum_dot;
    r_pred_sq[tid] = sum_pred_sq;
    r_target_sq[tid] = sum_target_sq;
    __syncthreads();

    // Perform tree-based reduction over the block
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            r_dot[tid] += r_dot[tid + s];
            r_pred_sq[tid] += r_pred_sq[tid + s];
            r_target_sq[tid] += r_target_sq[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 computes the final cosine similarity loss for this row
    if (tid == 0) {
        const float eps = 1e-8f;
        float norm_pred = sqrtf(r_pred_sq[0]);
        float norm_target = sqrtf(r_target_sq[0]);
        float denominator = norm_pred * norm_target;
        denominator = fmaxf(denominator, eps);
        float cos_sim = r_dot[0] / denominator;
        atomicAdd(output, 1.0f - cos_sim);
    }
}

// Host function to launch the kernel
torch::Tensor cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    const int block_size = 256;
    // Total shared memory: 2*TILE_SIZE for input tiles + 3*block_size for reduction arrays
    size_t shared_mem = (2 * TILE_SIZE + 3 * block_size) * sizeof(float);

    // Launch one block per sample (row)
    cosine_similarity_loss_kernel_shared<<<N, block_size, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    output.div_(N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward with Shared Memory Tiling (CUDA)");
}
