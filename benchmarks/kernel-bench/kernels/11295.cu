#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle down
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void shfl_shared_reduction_cosine_loss_kernel(const float* __restrict__ predictions,
                                                          const float* __restrict__ targets,
                                                          float* output,
                                                          const int N,
                                                          const int D) {
    // Each block handles one row (sample)
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    float dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    const int base = row * D;
    // Accumulate partial sums
    for (int i = tid; i < D; i += blockDim.x) {
        float p = predictions[base + i];
        float t = targets[base + i];
        dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Intra-warp reduction using shuffle
    dot = warp_reduce_sum(dot);
    sum_pred_sq = warp_reduce_sum(sum_pred_sq);
    sum_target_sq = warp_reduce_sum(sum_target_sq);

    // Shared memory to hold warp-level results
    __shared__ float s_dot[32];  // assuming blockDim.x <= 1024 (max 32 warps)
    __shared__ float s_pred_sq[32];
    __shared__ float s_target_sq[32];

    int lane = tid % warpSize;
    int warp_id = tid / warpSize;

    // First lane in each warp writes its result to shared memory
    if (lane == 0) {
        s_dot[warp_id] = dot;
        s_pred_sq[warp_id] = sum_pred_sq;
        s_target_sq[warp_id] = sum_target_sq;
    }

    __syncthreads();

    // Final reduction: let the first warp reduce the results from all warps
    float final_dot = 0.0f;
    float final_pred_sq = 0.0f;
    float final_target_sq = 0.0f;

    // Use only the first warp for final reduction
    if (tid < (blockDim.x / warpSize)) {
        final_dot = s_dot[lane];
        final_pred_sq = s_pred_sq[lane];
        final_target_sq = s_target_sq[lane];
    } else {
        final_dot = 0.0f;
        final_pred_sq = 0.0f;
        final_target_sq = 0.0f;
    }

    if (tid < warpSize) {
        final_dot = warp_reduce_sum(final_dot);
        final_pred_sq = warp_reduce_sum(final_pred_sq);
        final_target_sq = warp_reduce_sum(final_target_sq);
        
        // Compute final cosine similarity loss in the first thread
        if (lane == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(final_pred_sq);
            float norm_target = sqrtf(final_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = (denominator > eps) ? denominator : eps;
            float cos_sim = final_dot / denominator;
            atomicAdd(output, (1.0f - cos_sim) / N);
        }
    }
}


torch::Tensor shfl_shared_reduction_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());

    // Launch one block per row
    const int block_size = 256;
    shfl_shared_reduction_cosine_loss_kernel<<<N, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shfl_shared_reduction_cosine_similarity_loss_forward, "Shfl Shared Reduction Cosine Similarity Loss Forward (CUDA)");
}
