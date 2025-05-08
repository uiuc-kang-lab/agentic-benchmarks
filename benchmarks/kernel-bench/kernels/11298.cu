#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void ldg_aligned_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                          const float* __restrict__ targets,
                                                          float* output,
                                                          const int N,
                                                          const int D) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    
    const int vec_size = 4;
    const int D_aligned = D - (D % vec_size);
    
    const float4* pred_vec = reinterpret_cast<const float4*>(predictions + row * D);
    const float4* target_vec = reinterpret_cast<const float4*>(targets + row * D);
    
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    #pragma unroll 4
    for (int i = tid; i < D_aligned/vec_size; i += blockDim.x) {
        float4 pred = __ldg(&pred_vec[i]);
        float4 tgt = __ldg(&target_vec[i]);
        
        sum_dot += pred.x * tgt.x + pred.y * tgt.y + 
                   pred.z * tgt.z + pred.w * tgt.w;
        sum_pred_sq += pred.x * pred.x + pred.y * pred.y + 
                       pred.z * pred.z + pred.w * pred.w;
        sum_target_sq += tgt.x * tgt.x + tgt.y * tgt.y + 
                        tgt.z * tgt.z + tgt.w * tgt.w;
    }

    #pragma unroll
    for (int i = D_aligned + tid; i < D; i += blockDim.x) {
        float p = __ldg(&predictions[row * D + i]);
        float t = __ldg(&targets[row * D + i]);
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    sum_dot = warp_reduce_sum(sum_dot);
    sum_pred_sq = warp_reduce_sum(sum_pred_sq);
    sum_target_sq = warp_reduce_sum(sum_target_sq);

    __shared__ float s_dot[32];
    __shared__ float s_pred_sq[32];
    __shared__ float s_target_sq[32];

    if (lane_id == 0) {
        s_dot[warp_id] = sum_dot;
        s_pred_sq[warp_id] = sum_pred_sq;
        s_target_sq[warp_id] = sum_target_sq;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < warps_per_block) {
        sum_dot = s_dot[lane_id];
        sum_pred_sq = s_pred_sq[lane_id];
        sum_target_sq = s_target_sq[lane_id];

        sum_dot = warp_reduce_sum(sum_dot);
        sum_pred_sq = warp_reduce_sum(sum_pred_sq);
        sum_target_sq = warp_reduce_sum(sum_target_sq);

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

torch::Tensor ldg_aligned_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    
    const int block_size = 512;  // Using larger block size for better performance
    
    ldg_aligned_cosine_similarity_loss_kernel<<<N, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ldg_aligned_cosine_similarity_loss_forward, "LDG Aligned Cosine Similarity Loss Forward (CUDA)");
}