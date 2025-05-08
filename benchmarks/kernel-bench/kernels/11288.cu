#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void hybrid_cosine_similarity_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    const int N,
    const int D
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_WARP = 32;
    constexpr int VECTOR_LOAD_SIZE = 4;  // Process 4 elements per thread at once
    
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int lane_id = threadIdx.x % THREADS_PER_WARP;
    const int row = blockIdx.x;
    
    const float4* pred_row_vec = reinterpret_cast<const float4*>(predictions + row * D);
    const float4* target_row_vec = reinterpret_cast<const float4*>(targets + row * D);
    
    const int elements_per_warp = (D/VECTOR_LOAD_SIZE + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int start_idx = warp_id * elements_per_warp;
    const int end_idx = min(start_idx + elements_per_warp, D/VECTOR_LOAD_SIZE);

    float dot_sum = 0.0f;
    float pred_sq_sum = 0.0f;
    float target_sq_sum = 0.0f;

    // Vectorized memory access
    for (int idx = start_idx + lane_id; idx < end_idx; idx += THREADS_PER_WARP) {
        float4 pred_vec = pred_row_vec[idx];
        float4 target_vec = target_row_vec[idx];
        
        // Process 4 elements at once
        dot_sum += pred_vec.x * target_vec.x + pred_vec.y * target_vec.y + 
                  pred_vec.z * target_vec.z + pred_vec.w * target_vec.w;
        pred_sq_sum += pred_vec.x * pred_vec.x + pred_vec.y * pred_vec.y + 
                      pred_vec.z * pred_vec.z + pred_vec.w * pred_vec.w;
        target_sq_sum += target_vec.x * target_vec.x + target_vec.y * target_vec.y + 
                        target_vec.z * target_vec.z + target_vec.w * target_vec.w;
    }

    // Handle remaining elements
    const int remaining_start = (D / VECTOR_LOAD_SIZE) * VECTOR_LOAD_SIZE;
    if (remaining_start < D && threadIdx.x < (D - remaining_start)) {
        int idx = remaining_start + threadIdx.x;
        float pred = predictions[row * D + idx];
        float target = targets[row * D + idx];
        dot_sum += pred * target;
        pred_sq_sum += pred * pred;
        target_sq_sum += target * target;
    }

    // Warp-level reduction
    dot_sum = warpReduceSum(dot_sum);
    pred_sq_sum = warpReduceSum(pred_sq_sum);
    target_sq_sum = warpReduceSum(target_sq_sum);

    __shared__ float s_dot[WARPS_PER_BLOCK];
    __shared__ float s_pred_sq[WARPS_PER_BLOCK];
    __shared__ float s_target_sq[WARPS_PER_BLOCK];

    if (lane_id == 0) {
        s_dot[warp_id] = dot_sum;
        s_pred_sq[warp_id] = pred_sq_sum;
        s_target_sq[warp_id] = target_sq_sum;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < WARPS_PER_BLOCK) {
        dot_sum = s_dot[lane_id];
        pred_sq_sum = s_pred_sq[lane_id];
        target_sq_sum = s_target_sq[lane_id];

        dot_sum = warpReduceSum(dot_sum);
        pred_sq_sum = warpReduceSum(pred_sq_sum);
        target_sq_sum = warpReduceSum(target_sq_sum);

        if (lane_id == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(pred_sq_sum);
            float norm_target = sqrtf(target_sq_sum);
            float denominator = fmaxf(norm_pred * norm_target, eps);
            float cos_sim = dot_sum / denominator;
            atomicAdd(output, (1.0f - cos_sim) / N);
        }
    }
}

torch::Tensor hybrid_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");
    TORCH_CHECK(predictions.is_contiguous(), "predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    const int threads_per_block = 256;  // 8 warps * 32 threads

    hybrid_cosine_similarity_loss_kernel<<<N, threads_per_block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}