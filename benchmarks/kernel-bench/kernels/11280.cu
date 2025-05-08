#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void warp_aligned_cosine_loss_kernel(const float* __restrict__ predictions,
                                               const float* __restrict__ targets,
                                               float* output,
                                               int N,
                                               int D) {
    extern __shared__ float s_data[];
    const int warp_size = 32;
    const int warps_per_block = blockDim.x / warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int lane_id = threadIdx.x % warp_size;
    const int row = blockIdx.x;
    
    // Align data pointers for the current row
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;
    
    // Initialize accumulators
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;
    
    // Vectorized load and compute
    #pragma unroll 4
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float p = pred_row[i];
        float t = target_row[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }
    
    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        s_data[warp_id] = sum_dot;
        s_data[warp_id + warps_per_block] = sum_pred_sq;
        s_data[warp_id + 2 * warps_per_block] = sum_target_sq;
    }
    __syncthreads();
    
    // Final reduction using the first warp
    if (warp_id == 0) {
        sum_dot = (lane_id < warps_per_block) ? s_data[lane_id] : 0.0f;
        sum_pred_sq = (lane_id < warps_per_block) ? s_data[lane_id + warps_per_block] : 0.0f;
        sum_target_sq = (lane_id < warps_per_block) ? s_data[lane_id + 2 * warps_per_block] : 0.0f;
        
        // Warp-level reduction for final values
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset >>= 1) {
            sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
            sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
            sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
        }
        
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

torch::Tensor warp_aligned_cosine_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    
    // Ensure block size is multiple of warp size for aligned access
    const int block_size = 128; // 4 warps per block
    const int warps_per_block = block_size / 32;
    size_t shared_mem = 3 * warps_per_block * sizeof(float);

    warp_aligned_cosine_loss_kernel<<<N, block_size, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_aligned_cosine_loss_forward, "Warp Aligned Cosine Similarity Loss Forward (CUDA)");
}