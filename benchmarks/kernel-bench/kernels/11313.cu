#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__forceinline__ __device__ float warp_reduce(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void strided_cosine_similarity_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    const int N,
    const int D
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / warpSize;
    const int lane = tid % warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    
    // Use float4 for coalesced memory access
    const int vec_size = 4;
    const int D_vec = D / vec_size;
    const int D_remainder = D % vec_size;
    
    // Calculate optimal stride for the vectorized portion
    const int stride = (D_vec + blockDim.x - 1) / blockDim.x * vec_size;
    
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Process vectorized elements with stride
    const float4* pred_vec = reinterpret_cast<const float4*>(predictions + row * D);
    const float4* targ_vec = reinterpret_cast<const float4*>(targets + row * D);
    
    #pragma unroll 4
    for (int idx = tid; idx < D_vec; idx += blockDim.x) {
        float4 pred = pred_vec[idx];
        float4 targ = targ_vec[idx];
        
        sum_dot += pred.x * targ.x + pred.y * targ.y + 
                  pred.z * targ.z + pred.w * targ.w;
        sum_pred_sq += pred.x * pred.x + pred.y * pred.y + 
                      pred.z * pred.z + pred.w * pred.w;
        sum_target_sq += targ.x * targ.x + targ.y * targ.y + 
                        targ.z * targ.z + targ.w * targ.w;
    }

    // Handle remaining elements
    const int rem_start = D_vec * vec_size;
    #pragma unroll
    for (int idx = rem_start + tid; idx < D; idx += blockDim.x) {
        float pred = predictions[row * D + idx];
        float targ = targets[row * D + idx];
        sum_dot += pred * targ;
        sum_pred_sq += pred * pred;
        sum_target_sq += targ * targ;
    }

    // Warp-level reduction
    sum_dot = warp_reduce(sum_dot);
    sum_pred_sq = warp_reduce(sum_pred_sq);
    sum_target_sq = warp_reduce(sum_target_sq);

    // Block-level reduction using shared memory
    __shared__ float s_dot[32];        // One element per warp
    __shared__ float s_pred_sq[32];
    __shared__ float s_target_sq[32];

    if (lane == 0) {
        s_dot[wid] = sum_dot;
        s_pred_sq[wid] = sum_pred_sq;
        s_target_sq[wid] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction by first warp
    if (tid < warps_per_block) {
        sum_dot = (tid < warps_per_block) ? s_dot[tid] : 0.0f;
        sum_pred_sq = (tid < warps_per_block) ? s_pred_sq[tid] : 0.0f;
        sum_target_sq = (tid < warps_per_block) ? s_target_sq[tid] : 0.0f;

        sum_dot = warp_reduce(sum_dot);
        sum_pred_sq = warp_reduce(sum_pred_sq);
        sum_target_sq = warp_reduce(sum_target_sq);

        if (tid == 0) {
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

torch::Tensor strided_cosine_similarity_loss_forward(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    
    // Select block size based on dimension size
    const int block_size = (D <= 256) ? 256 : 512;
    
    strided_cosine_similarity_loss_kernel<<<N, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &strided_cosine_similarity_loss_forward, "Strided Cosine Similarity Loss Forward (CUDA)");
}