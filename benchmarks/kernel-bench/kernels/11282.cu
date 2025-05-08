#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void balanced_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                       const float* __restrict__ targets,
                                                       float* output,
                                                       const int N,
                                                       const int D) {
    // Use 8 warps per block for optimal occupancy
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_WARP = 32;
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int lane_id = threadIdx.x % THREADS_PER_WARP;
    const int row = blockIdx.x;
    
    // Calculate starting positions for this row
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;
    
    // Each warp processes a chunk of the vector
    const int elements_per_warp = (D + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int start_idx = warp_id * elements_per_warp;
    const int end_idx = min(start_idx + elements_per_warp, D);

    // Process elements within each warp
    float dot_sum = 0.0f;
    float pred_sq_sum = 0.0f;
    float target_sq_sum = 0.0f;

    // Coalesced memory access pattern within warps
    for (int idx = start_idx + lane_id; idx < end_idx; idx += THREADS_PER_WARP) {
        float pred = pred_row[idx];
        float target = target_row[idx];
        dot_sum += pred * target;
        pred_sq_sum += pred * pred;
        target_sq_sum += target * target;
    }

    // Warp-level reduction using shuffle
    dot_sum = warpReduceSum(dot_sum);
    pred_sq_sum = warpReduceSum(pred_sq_sum);
    target_sq_sum = warpReduceSum(target_sq_sum);

    // Shared memory for inter-warp reduction
    __shared__ float s_dot[WARPS_PER_BLOCK];
    __shared__ float s_pred_sq[WARPS_PER_BLOCK];
    __shared__ float s_target_sq[WARPS_PER_BLOCK];

    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        s_dot[warp_id] = dot_sum;
        s_pred_sq[warp_id] = pred_sq_sum;
        s_target_sq[warp_id] = target_sq_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        dot_sum = (lane_id < WARPS_PER_BLOCK) ? s_dot[lane_id] : 0.0f;
        pred_sq_sum = (lane_id < WARPS_PER_BLOCK) ? s_pred_sq[lane_id] : 0.0f;
        target_sq_sum = (lane_id < WARPS_PER_BLOCK) ? s_target_sq[lane_id] : 0.0f;

        // Final warp-level reduction
        dot_sum = warpReduceSum(dot_sum);
        pred_sq_sum = warpReduceSum(pred_sq_sum);
        target_sq_sum = warpReduceSum(target_sq_sum);

        if (lane_id == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(pred_sq_sum);
            float norm_target = sqrtf(target_sq_sum);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = dot_sum / denominator;
            atomicAdd(output, (1.0f - cos_sim) / N);
        }
    }
}

torch::Tensor balanced_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());

    // Use 8 warps (256 threads) per block
    const int threads_per_block = 256;
    
    balanced_cosine_similarity_loss_kernel<<<N, threads_per_block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &balanced_cosine_similarity_loss_forward, "Balanced Cosine Similarity Loss Forward (CUDA)");
}