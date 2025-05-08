#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                              const float* __restrict__ targets,
                                              float* output,
                                              int N,
                                              int D) {
    const unsigned int FULL_MASK = 0xffffffff;
    const int warpSize = 32;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    int warps_per_block = blockDim.x / warpSize;
    
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;
    
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;
    
    // Coalesced memory access and parallel reduction
    for (int i = tid; i < D; i += blockDim.x) {
        float p = pred_row[i];
        float t = target_row[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum_dot += __shfl_down_sync(FULL_MASK, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(FULL_MASK, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(FULL_MASK, sum_target_sq, offset);
    }
    
    // First thread in each warp writes to shared memory
    extern __shared__ float s_data[];
    float* s_dot = s_data;
    float* s_pred_sq = s_data + warps_per_block;
    float* s_target_sq = s_pred_sq + warps_per_block;
    
    if (lane == 0) {
        s_dot[wid] = sum_dot;
        s_pred_sq[wid] = sum_pred_sq;
        s_target_sq[wid] = sum_target_sq;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < warpSize) {
        float block_dot = (tid < warps_per_block) ? s_dot[tid] : 0.0f;
        float block_pred_sq = (tid < warps_per_block) ? s_pred_sq[tid] : 0.0f;
        float block_target_sq = (tid < warps_per_block) ? s_target_sq[tid] : 0.0f;
        
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            block_dot += __shfl_down_sync(FULL_MASK, block_dot, offset);
            block_pred_sq += __shfl_down_sync(FULL_MASK, block_pred_sq, offset);
            block_target_sq += __shfl_down_sync(FULL_MASK, block_target_sq, offset);
        }
        
        if (tid == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(block_pred_sq);
            float norm_target = sqrtf(block_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            
            float cos_sim = block_dot / denominator;
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
    const int warps_per_block = block_size / 32;
    size_t shared_mem = 3 * warps_per_block * sizeof(float);

    cosine_similarity_loss_kernel<<<N, block_size, shared_mem>>>(
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