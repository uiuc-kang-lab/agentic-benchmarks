#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ void warp_reduce_triple(float& dot, float& pred_sq, float& target_sq) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
        pred_sq += __shfl_down_sync(0xffffffff, pred_sq, offset);
        target_sq += __shfl_down_sync(0xffffffff, target_sq, offset);
    }
}

__global__ void warp_optimized_cosine_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    const int N,
    const int D) {
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    const int num_warps = blockDim.x / warpSize;
    
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Process elements with stride
    for (int i = tid; i < D; i += blockDim.x) {
        float p = predictions[row * D + i];
        float t = targets[row * D + i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // First level reduction within each warp
    warp_reduce_triple(sum_dot, sum_pred_sq, sum_target_sq);

    // Inter-warp reduction using first thread of each warp
    if (lane_id == 0) {
        // Broadcast warp results to first warp
        float warp_dot = sum_dot;
        float warp_pred_sq = sum_pred_sq;
        float warp_target_sq = sum_target_sq;
        
        if (warp_id == 0) {
            // First warp collects results
            for (int w = 1; w < num_warps; w++) {
                sum_dot += __shfl_sync(0xffffffff, warp_dot, w * warpSize);
                sum_pred_sq += __shfl_sync(0xffffffff, warp_pred_sq, w * warpSize);
                sum_target_sq += __shfl_sync(0xffffffff, warp_target_sq, w * warpSize);
            }

            // Final reduction in first warp
            if (lane_id == 0) {
                const float eps = 1e-8f;
                float norm_pred = sqrtf(sum_pred_sq);
                float norm_target = sqrtf(sum_target_sq);
                float denominator = norm_pred * norm_target;
                denominator = fmaxf(denominator, eps);
                float cos_sim = sum_dot / denominator;
                atomicAdd(output, (1.0f - cos_sim) / N);
            }
        }
    }
}

torch::Tensor warp_optimized_cosine_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    const int block_size = 256;  // 8 warps per block
    
    warp_optimized_cosine_loss_kernel<<<N, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_optimized_cosine_loss_forward, "Warp Optimized Cosine Loss Forward (CUDA)");
}