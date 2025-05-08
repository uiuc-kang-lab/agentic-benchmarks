#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void distributed_cosine_similarity_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ block_results,
    const int N,
    const int D) {
    
    extern __shared__ float shared[];
    float* s_dot = shared;
    float* s_pred_sq = &shared[blockDim.x];
    float* s_target_sq = &shared[2 * blockDim.x];
    
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;
    
    // Align to float4 boundary
    const int vec_size = 4;
    const int D_aligned = D - (D % vec_size);
    
    const float4* pred_vec = reinterpret_cast<const float4*>(predictions + row * D);
    const float4* target_vec = reinterpret_cast<const float4*>(targets + row * D);
    
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Vectorized processing
    #pragma unroll 4
    for (int i = tid; i < D_aligned/vec_size; i += blockDim.x) {
        float4 pred = pred_vec[i];
        float4 tgt = target_vec[i];
        
        sum_dot += pred.x * tgt.x + pred.y * tgt.y + 
                   pred.z * tgt.z + pred.w * tgt.w;
        sum_pred_sq += pred.x * pred.x + pred.y * pred.y + 
                      pred.z * pred.z + pred.w * pred.w;
        sum_target_sq += tgt.x * tgt.x + tgt.y * tgt.y + 
                        tgt.z * tgt.z + tgt.w * tgt.w;
    }

    // Handle remaining elements
    for (int i = D_aligned + tid; i < D; i += blockDim.x) {
        float p = predictions[row * D + i];
        float t = targets[row * D + i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Warp-level reduction
    sum_dot = warp_reduce_sum(sum_dot);
    sum_pred_sq = warp_reduce_sum(sum_pred_sq);
    sum_target_sq = warp_reduce_sum(sum_target_sq);

    if (lane_id == 0) {
        s_dot[warp_id] = sum_dot;
        s_pred_sq[warp_id] = sum_pred_sq;
        s_target_sq[warp_id] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction in first warp
    if (tid < warpSize) {
        const int num_warps = blockDim.x / warpSize;
        sum_dot = (tid < num_warps) ? s_dot[tid] : 0.0f;
        sum_pred_sq = (tid < num_warps) ? s_pred_sq[tid] : 0.0f;
        sum_target_sq = (tid < num_warps) ? s_target_sq[tid] : 0.0f;

        sum_dot = warp_reduce_sum(sum_dot);
        sum_pred_sq = warp_reduce_sum(sum_pred_sq);
        sum_target_sq = warp_reduce_sum(sum_target_sq);

        if (tid == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(sum_pred_sq);
            float norm_target = sqrtf(sum_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = sum_dot / denominator;
            
            // Store result for this block - no atomic needed here
            block_results[row] = 1.0f - cos_sim;
        }
    }
}

__global__ void final_reduction_kernel(float* block_results, float* output, int N) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += block_results[i];
    }
    
    sum = warp_reduce_sum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum / N);
    }
}

torch::Tensor distributed_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    auto block_results = torch::empty({N}, predictions.options());
    
    const int block_size = 256;
    const size_t shared_mem_size = 3 * block_size * sizeof(float);

    distributed_cosine_similarity_loss_kernel<<<N, block_size, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        N,
        D
    );

    final_reduction_kernel<<<1, 256>>>(
        block_results.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &distributed_cosine_similarity_loss_forward, "Distributed Cosine Similarity Loss Forward (CUDA)");
}