#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel assigns one warp (32 threads) per row,
// avoiding the need for __syncthreads() since warp-level intrinsics
// ensure synchronization within the warp.
__global__ void warp_row_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                          const float* __restrict__ targets,
                                                          float* output,
                                                          const int N,
                                                          const int D) {
    // Each block processes one row
    int row = blockIdx.x;
    // Use exactly one warp per block
    int lane = threadIdx.x;  // Assumes blockDim.x == 32

    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float dot = 0.0f;
    float pred_sq = 0.0f;
    float target_sq = 0.0f;

    // Each thread processes a strided set of elements
    for (int i = lane; i < D; i += 32) {
        float p = pred_row[i];
        float t = target_row[i];
        dot += p * t;
        pred_sq += p * p;
        target_sq += t * t;
    }

    // Perform warp-level reduction using shuffle intrinsics
    for (int offset = 16; offset > 0; offset /= 2) {
        dot      += __shfl_down_sync(0xffffffff, dot, offset);
        pred_sq  += __shfl_down_sync(0xffffffff, pred_sq, offset);
        target_sq+= __shfl_down_sync(0xffffffff, target_sq, offset);
    }

    // Only lane 0 has the complete sum for this row
    if (lane == 0) {
        const float eps = 1e-8f;
        float norm_pred = sqrtf(pred_sq);
        float norm_target = sqrtf(target_sq);
        float denominator = norm_pred * norm_target;
        denominator = fmaxf(denominator, eps);
        float cosine_sim = dot / denominator;
        float row_loss = 1.0f - cosine_sim;
        // Atomic addition to accumulate loss across rows, averaging by N
        atomicAdd(output, row_loss / static_cast<float>(N));
    }
}


torch::Tensor warp_row_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());

    // Launch one block per row, with a single warp (32 threads) per block
    dim3 grid(N);
    dim3 block(32);
    
    warp_row_cosine_similarity_loss_kernel<<<grid, block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_row_cosine_similarity_loss_forward, "Warp Row Cosine Similarity Loss Forward (CUDA)");
}
