#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define tile size for shared memory tiling
#define TILE_SIZE 512

// Kernel to compute cosine similarity loss using shared memory tiling
__global__ void cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                const float* __restrict__ targets,
                                                float* output,
                                                int N,
                                                int D) {
    // Dynamically allocated shared memory layout:
    // First: tile of predictions (TILE_SIZE floats)
    // Second: tile of targets (TILE_SIZE floats)
    // Third: reduction arrays for dot product (blockDim.x floats)
    // Fourth: reduction array for pred square sum (blockDim.x floats)
    // Fifth: reduction array for target square sum (blockDim.x floats)
    extern __shared__ float shared[];
    float* tile_pred    = shared;                              // size: TILE_SIZE
    float* tile_target  = tile_pred + TILE_SIZE;               // size: TILE_SIZE
    float* s_dot        = tile_target + TILE_SIZE;             // size: blockDim.x
    float* s_pred_sq    = s_dot + blockDim.x;                  // size: blockDim.x
    float* s_target_sq  = s_pred_sq + blockDim.x;              // size: blockDim.x

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    // Process the row in tiles of TILE_SIZE elements
    for (int tile_offset = 0; tile_offset < D; tile_offset += TILE_SIZE) {
        // Load a tile of predictions and targets into shared memory
        for (int j = tid; j < TILE_SIZE; j += blockDim.x) {
            int idx = tile_offset + j;
            if (idx < D) {
                tile_pred[j] = pred_row[idx];
                tile_target[j] = target_row[idx];
            } else {
                tile_pred[j] = 0.0f;
                tile_target[j] = 0.0f;
            }
        }
        __syncthreads();

        // Compute partial sums for the current tile
        for (int j = tid; j < TILE_SIZE; j += blockDim.x) {
            int idx = tile_offset + j;
            if (idx < D) {
                float p = tile_pred[j];
                float t = tile_target[j];
                sum_dot += p * t;
                sum_pred_sq += p * p;
                sum_target_sq += t * t;
            }
        }
        __syncthreads();
    }

    // Reduce partial sums within the block using shared memory
    s_dot[tid] = sum_dot;
    s_pred_sq[tid] = sum_pred_sq;
    s_target_sq[tid] = sum_target_sq;
    __syncthreads();

    // Binary tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_dot[tid] += s_dot[tid + s];
            s_pred_sq[tid] += s_pred_sq[tid + s];
            s_target_sq[tid] += s_target_sq[tid + s];
        }
        __syncthreads();
    }

    // Only one thread per block computes the final cosine similarity loss contribution
    if (tid == 0) {
        const float eps = 1e-8f;
        float norm_pred = sqrtf(s_pred_sq[0]);
        float norm_target = sqrtf(s_target_sq[0]);
        float denominator = norm_pred * norm_target;
        denominator = fmaxf(denominator, eps);
        float cos_sim = s_dot[0] / denominator;
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
    
    // Shared memory size: 2*TILE_SIZE for tiling + 3*block_size for reductions
    size_t shared_mem = (2 * TILE_SIZE + 3 * block_size) * sizeof(float);

    cosine_similarity_loss_kernel<<<N, block_size, shared_mem>>>(
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
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward (CUDA) with shared memory tiling");
}
