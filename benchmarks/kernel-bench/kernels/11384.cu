#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel using warp-only reduction. Launch one warp (32 threads) per row.
// This eliminates the need for shared memory for reduction, leveraging __shfl_down_sync for intra-warp reductions.
__global__ void cosine_similarity_loss_kernel_warp_only(const float* __restrict__ predictions,
                                                          const float* __restrict__ targets,
                                                          float* output,
                                                          int N,
                                                          int D) {
    // Each block processes one row; blockDim.x is fixed to 32 (one warp per block)
    int row = blockIdx.x;
    int tid = threadIdx.x;  // 0 to 31

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Precompute row offset to avoid repeated multiplication
    int row_offset = row * D;
    // Each thread processes a subset of the D elements with stride equal to warp size (32)
    for (int j = tid; j < D; j += 32) {
        float p = __ldg(&predictions[row_offset + j]);
        float t = __ldg(&targets[row_offset + j]);
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Perform warp-level reduction using shuffle operations
    // The mask 0xffffffff covers all 32 threads in the warp
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_dot      += __shfl_down_sync(0xffffffff, sum_dot, offset);
        sum_pred_sq  += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
        sum_target_sq+= __shfl_down_sync(0xffffffff, sum_target_sq, offset);
    }

    // The first lane in the warp now holds the reduced sums
    if (tid == 0) {
        const float eps = 1e-8f;
        float norm_pred = sqrtf(sum_pred_sq);
        float norm_target = sqrtf(sum_target_sq);
        float denominator = norm_pred * norm_target;
        denominator = fmaxf(denominator, eps);
        float cos_sim = sum_dot / denominator;
        // Each block computes one row; atomically accumulate the loss
        atomicAdd(output, 1.0f - cos_sim);
    }
}

// Host function: Launch one warp per row (blockDim.x = 32).
// This kernel avoids shared memory usage for reductions by relying solely on warp-level primitives.

torch::Tensor cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);
    
    auto output = torch::zeros({1}, predictions.options());

    // Launch one block per row with exactly 32 threads (one warp per block)
    const int block_size = 32;
    cosine_similarity_loss_kernel_warp_only<<<N, block_size>>>(
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
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward with warp-only reduction (CUDA)");
}
