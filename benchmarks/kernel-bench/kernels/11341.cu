#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template<int BLOCK_SIZE = 256, int VECTOR_SIZE = 4>
__global__ void cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                            const float* __restrict__ targets,
                                            float* output,
                                            const int N,
                                            const int D) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    constexpr int warpSize = 32;
    const int lane = tid & (warpSize - 1);
    const int warp_id = tid / warpSize;

    const float4* pred_row_vec = reinterpret_cast<const float4*>(predictions + row * D);
    const float4* target_row_vec = reinterpret_cast<const float4*>(targets + row * D);
    
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    const int D_aligned = D / VECTOR_SIZE * VECTOR_SIZE;
    
    #pragma unroll 4
    for (int i = tid * VECTOR_SIZE; i < D_aligned; i += BLOCK_SIZE * VECTOR_SIZE) {
        const float4 pred_vec = __ldg(&pred_row_vec[i/VECTOR_SIZE]);
        const float4 target_vec = __ldg(&target_row_vec[i/VECTOR_SIZE]);
        
        sum_dot += pred_vec.x * target_vec.x + 
                  pred_vec.y * target_vec.y + 
                  pred_vec.z * target_vec.z + 
                  pred_vec.w * target_vec.w;
                  
        sum_pred_sq += pred_vec.x * pred_vec.x + 
                      pred_vec.y * pred_vec.y + 
                      pred_vec.z * pred_vec.z + 
                      pred_vec.w * pred_vec.w;
                      
        sum_target_sq += target_vec.x * target_vec.x + 
                        target_vec.y * target_vec.y + 
                        target_vec.z * target_vec.z + 
                        target_vec.w * target_vec.w;
    }

    #pragma unroll
    for (int i = D_aligned + tid; i < D; i += BLOCK_SIZE) {
        const float pred = __ldg(&predictions[row * D + i]);
        const float target = __ldg(&targets[row * D + i]);
        sum_dot += pred * target;
        sum_pred_sq += pred * pred;
        sum_target_sq += target * target;
    }

    constexpr unsigned int FULL_MASK = 0xffffffff;
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        sum_dot += __shfl_down_sync(FULL_MASK, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(FULL_MASK, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(FULL_MASK, sum_target_sq, offset);
    }

    extern __shared__ float shared[];
    constexpr int nWarps = BLOCK_SIZE / warpSize;
    
    if (lane == 0) {
        shared[warp_id] = sum_dot;
        shared[warp_id + nWarps] = sum_pred_sq;
        shared[warp_id + 2 * nWarps] = sum_target_sq;
    }
    __syncthreads();

    if (tid < warpSize) {
        float final_dot = (tid < nWarps) ? shared[tid] : 0.0f;
        float final_pred_sq = (tid < nWarps) ? shared[tid + nWarps] : 0.0f;
        float final_target_sq = (tid < nWarps) ? shared[tid + 2 * nWarps] : 0.0f;

        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            final_dot += __shfl_down_sync(FULL_MASK, final_dot, offset);
            final_pred_sq += __shfl_down_sync(FULL_MASK, final_pred_sq, offset);
            final_target_sq += __shfl_down_sync(FULL_MASK, final_target_sq, offset);
        }

        if (lane == 0) {
            constexpr float eps = 1e-8f;
            const float norm_pred = sqrtf(final_pred_sq);
            const float norm_target = sqrtf(final_target_sq);
            const float denominator = fmaxf(norm_pred * norm_target, eps);
            const float cos_sim = final_dot / denominator;
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
    
    const int N = predictions.size(0);
    const int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    constexpr int BLOCK_SIZE = 256;
    constexpr int nWarps = BLOCK_SIZE / 32;
    const size_t shared_mem = 3 * nWarps * sizeof(float);

    cosine_similarity_loss_kernel<BLOCK_SIZE><<<N, BLOCK_SIZE, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    output.div_(N);
    return output;
}