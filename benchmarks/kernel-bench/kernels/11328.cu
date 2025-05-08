#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel using stride loops and warp-level reductions with uniform control flow to minimize divergent branching
__global__ void cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                               const float* __restrict__ targets,
                                               float* output,
                                               int N,
                                               int D) {
    // Each block handles one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warpSize = 32;
    int lane = tid & (warpSize - 1);
    int warp_id = tid / warpSize;

    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Process the row with coalesced memory access pattern
    // Each thread processes consecutive elements within a block-wide chunk
    const int items_per_block = (D + blockSize - 1) / blockSize;
    const int block_start = 0;
    
    for (int block_offset = 0; block_offset < items_per_block; block_offset++) {
        const int i = block_offset * blockSize + tid;
        if (i < D) {  // Boundary check
            float p = pred_row[i];
            float t = target_row[i];
            sum_dot += p * t;
            sum_pred_sq += p * p;
            sum_target_sq += t * t;
        }
    }

    // Intra-warp reduction using warp shuffle; all lanes execute the same instructions
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_dot       += __shfl_down_sync(mask, sum_dot, offset);
        sum_pred_sq   += __shfl_down_sync(mask, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(mask, sum_target_sq, offset);
    }

    // Shared memory to store per-warp results
    extern __shared__ float s[];  // layout: [0, nWarps): dot, [nWarps, 2*nWarps): pred_sq, [2*nWarps, 3*nWarps): target_sq
    int nWarps = (blockSize + warpSize - 1) / warpSize;
    if (lane == 0) {
        s[warp_id] = sum_dot;
        s[warp_id + nWarps] = sum_pred_sq;
        s[warp_id + 2 * nWarps] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction performed by the first warp in a uniform control flow manner
    float final_dot = 0.0f;
    float final_pred_sq = 0.0f;
    float final_target_sq = 0.0f;
    if (tid < warpSize) {
        // Use ternary operator to ensure all threads in the warp execute the same instructions
        final_dot = (tid < nWarps) ? s[tid] : 0.0f;
        final_pred_sq = (tid < nWarps) ? s[tid + nWarps] : 0.0f;
        final_target_sq = (tid < nWarps) ? s[tid + 2 * nWarps] : 0.0f;
        
        // Uniform warp-level reduction across the first warp
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            final_dot       += __shfl_down_sync(mask, final_dot, offset);
            final_pred_sq   += __shfl_down_sync(mask, final_pred_sq, offset);
            final_target_sq += __shfl_down_sync(mask, final_target_sq, offset);
        }
    }

    if (tid == 0) {
        const float eps = 1e-8f;
        float norm_pred = sqrtf(final_pred_sq);
        float norm_target = sqrtf(final_target_sq);
        float denominator = norm_pred * norm_target;
        denominator = fmaxf(denominator, eps);
        float cos_sim = final_dot / denominator;
        atomicAdd(output, 1.0f - cos_sim);
    }
}

// Host function to launch the CUDA kernel
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
    const int warpSize = 32; // Host-side definition
    int nWarps = (block_size + warpSize - 1) / warpSize;
    size_t shared_mem = 3 * nWarps * sizeof(float);

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
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward (CUDA optimized with uniform warp control)");
}
