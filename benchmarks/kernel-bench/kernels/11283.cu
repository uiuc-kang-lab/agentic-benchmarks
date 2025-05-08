#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ void warpReduceSum3(float& val1, float& val2, float& val3) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val1 += __shfl_down_sync(0xffffffff, val1, offset);
        val2 += __shfl_down_sync(0xffffffff, val2, offset);
        val3 += __shfl_down_sync(0xffffffff, val3, offset);
    }
}

__global__ void warp_optimized_cosine_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int N,
    const int D
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / warpSize;
    const int lane = tid % warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    
    __shared__ float s_dot[32];
    __shared__ float s_pred_sq[32];
    __shared__ float s_target_sq[32];
    
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;
    
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;
    
    for (int i = tid; i < D; i += blockDim.x) {
        const float p = pred_row[i];
        const float t = target_row[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }
    
    warpReduceSum3(sum_dot, sum_pred_sq, sum_target_sq);
    
    if (lane == 0) {
        s_dot[wid] = sum_dot;
        s_pred_sq[wid] = sum_pred_sq;
        s_target_sq[wid] = sum_target_sq;
    }
    __syncthreads();
    
    if (wid == 0) {
        sum_dot = (lane < warps_per_block) ? s_dot[lane] : 0.0f;
        sum_pred_sq = (lane < warps_per_block) ? s_pred_sq[lane] : 0.0f;
        sum_target_sq = (lane < warps_per_block) ? s_target_sq[lane] : 0.0f;
        
        warpReduceSum3(sum_dot, sum_pred_sq, sum_target_sq);
        
        if (lane == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(sum_pred_sq);
            float norm_target = sqrtf(sum_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            
            float cos_sim = sum_dot / denominator;
            float loss = 1.0f - cos_sim;
            atomicAdd(output, loss / N);
        }
    }
}

torch::Tensor warp_optimized_cosine_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");
    
    const int N = predictions.size(0);
    const int D = predictions.size(1);
    
    auto output = torch::zeros({1}, predictions.options());
    
    const int block_size = 256;
    
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