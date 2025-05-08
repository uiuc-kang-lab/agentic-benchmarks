#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle instructions
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Combined CUDA kernel for cosine similarity loss that leverages coalesced vectorized loads, loop unrolling,
// and efficient warp-level and block-level reductions.
__global__ void efficient_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                         const float* __restrict__ targets,
                                                         float* output,
                                                         const int N,
                                                         const int D) {
    // Each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;  // expected to be 512
    int warps_per_block = blockSize / warpSize;

    // Vectorized load parameters
    const int vec_size = 4;
    int D_aligned = (D / vec_size) * vec_size;
    int num_vec = D_aligned / vec_size;

    // Cast pointers for coalesced float4 loads
    const float4* pred_vec = reinterpret_cast<const float4*>(predictions + row * D);
    const float4* target_vec = reinterpret_cast<const float4*>(targets + row * D);

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Process the vectorized portion with loop unrolling for improved throughput
    #pragma unroll 4
    for (int i = tid; i < num_vec; i += blockSize) {
        float4 p = pred_vec[i];
        float4 t = target_vec[i];
        sum_dot += p.x * t.x + p.y * t.y + p.z * t.z + p.w * t.w;
        sum_pred_sq += p.x * p.x + p.y * p.y + p.z * p.z + p.w * p.w;
        sum_target_sq += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
    }

    // Process remaining elements if D is not a multiple of vec_size
    for (int i = D_aligned + tid; i < D; i += blockSize) {
        float p = predictions[row * D + i];
        float t = targets[row * D + i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Intra-warp reduction
    sum_dot = warp_reduce_sum(sum_dot);
    sum_pred_sq = warp_reduce_sum(sum_pred_sq);
    sum_target_sq = warp_reduce_sum(sum_target_sq);

    // Use shared memory to reduce across warps
    __shared__ float s_dot[32];
    __shared__ float s_pred_sq[32];
    __shared__ float s_target_sq[32];

    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) {
        s_dot[warp_id] = sum_dot;
        s_pred_sq[warp_id] = sum_pred_sq;
        s_target_sq[warp_id] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction: only threads corresponding to warp leaders perform this
    if (tid < warps_per_block) {
        sum_dot = s_dot[tid];
        sum_pred_sq = s_pred_sq[tid];
        sum_target_sq = s_target_sq[tid];
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
            // Accumulate loss in a numerically safe manner
            atomicAdd(output, (1.0f - cos_sim) / N);
        }
    }
}

// Interface function called from PyTorch
torch::Tensor efficient_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    const int block_size = 512; // Optimal block size for coalesced loads and warp reduction
    
    efficient_cosine_similarity_loss_kernel<<<N, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &efficient_cosine_similarity_loss_forward, "Efficient Cosine Similarity Loss Forward (CUDA)");
}
