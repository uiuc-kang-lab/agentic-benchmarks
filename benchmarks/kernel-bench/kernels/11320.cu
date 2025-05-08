#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void cosine_similarity_loss_kernel(const float4* __restrict__ predictions4,
                                            const float4* __restrict__ targets4,
                                            float* output,
                                            int N,
                                            int D) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane = tid % warpSize;
    const int num_warps = blockDim.x / warpSize;
    
    // Align D to float4 boundary
    const int D4 = D / 4;
    
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Use float4 for vectorized memory access
    for (int i = tid; i < D4; i += blockDim.x) {
        float4 p4 = predictions4[row * D4 + i];
        float4 t4 = targets4[row * D4 + i];
        
        sum_dot += p4.x * t4.x + p4.y * t4.y + p4.z * t4.z + p4.w * t4.w;
        sum_pred_sq += p4.x * p4.x + p4.y * p4.y + p4.z * p4.z + p4.w * p4.w;
        sum_target_sq += t4.x * t4.x + t4.y * t4.y + t4.z * t4.z + t4.w * t4.w;
    }

    // Handle remaining elements
    const float* predictions = (const float*)predictions4;
    const float* targets = (const float*)targets4;
    for (int i = D4 * 4 + tid; i < D; i += blockDim.x) {
        float p = predictions[row * D + i];
        float t = targets[row * D + i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // First warp reduction
    sum_dot = warp_reduce_sum(sum_dot);
    sum_pred_sq = warp_reduce_sum(sum_pred_sq);
    sum_target_sq = warp_reduce_sum(sum_target_sq);

    // Use padding to avoid bank conflicts
    __shared__ float shared_results[3 * 33];  // Added padding
    if (lane == 0) {
        shared_results[warp_id] = sum_dot;
        shared_results[num_warps + warp_id] = sum_pred_sq;
        shared_results[2 * num_warps + warp_id] = sum_target_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        if (tid < num_warps) {
            sum_dot = shared_results[lane];
            sum_pred_sq = shared_results[num_warps + lane];
            sum_target_sq = shared_results[2 * num_warps + lane];
        } else {
            sum_dot = 0.0f;
            sum_pred_sq = 0.0f;
            sum_target_sq = 0.0f;
        }

        // Final warp reduction
        sum_dot = warp_reduce_sum(sum_dot);
        sum_pred_sq = warp_reduce_sum(sum_pred_sq);
        sum_target_sq = warp_reduce_sum(sum_target_sq);

        if (lane == 0) {
            const float eps = 1e-8f;
            float norm_pred = __fsqrt_rn(sum_pred_sq);  // Fast intrinsic square root
            float norm_target = __fsqrt_rn(sum_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);

            float cos_sim = sum_dot / denominator;
            atomicAdd(output, 1.0f - cos_sim);
        }
    }
}

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

    cosine_similarity_loss_kernel<<<N, block_size>>>(
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
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward (CUDA)");
}