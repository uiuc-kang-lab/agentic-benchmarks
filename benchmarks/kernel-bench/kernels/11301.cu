#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void coalesced_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                           const float* __restrict__ targets,
                                                           float* output,
                                                           int N,
                                                           int D) {
    // Each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x; // Expected to be 512

    // Use vectorized loads to ensure coalescing. We load 4 floats (16 bytes) at a time.
    const int vecSize = 4;
    int D_aligned = (D / vecSize) * vecSize;
    int numVec = D_aligned / vecSize;

    // Reinterpret the row pointers as float4* for aligned access
    const float4* predictions_vec = reinterpret_cast<const float4*>(predictions + row * D);
    const float4* targets_vec = reinterpret_cast<const float4*>(targets + row * D);

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Process vectorized portion; threads in a warp read consecutive float4 elements ensuring coalescing
    for (int i = tid; i < numVec; i += blockSize) {
        float4 p = predictions_vec[i];
        float4 t = targets_vec[i];
        sum_dot     += p.x * t.x + p.y * t.y + p.z * t.z + p.w * t.w;
        sum_pred_sq += p.x * p.x + p.y * p.y + p.z * p.z + p.w * p.w;
        sum_target_sq += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
    }

    // Process remaining elements if D is not divisible by 4
    for (int i = D_aligned + tid; i < D; i += blockSize) {
        float p = predictions[row * D + i];
        float t = targets[row * D + i];
        sum_dot     += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Perform warp-level reduction
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum_dot      += __shfl_down_sync(mask, sum_dot, offset);
        sum_pred_sq  += __shfl_down_sync(mask, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(mask, sum_target_sq, offset);
    }

    // Shared memory to hold results from each warp
    __shared__ float s_dot[512/32];
    __shared__ float s_pred_sq[512/32];
    __shared__ float s_target_sq[512/32];

    int warpId = tid / warpSize;
    if ((tid % warpSize) == 0) {
        s_dot[warpId] = sum_dot;
        s_pred_sq[warpId] = sum_pred_sq;
        s_target_sq[warpId] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction from each warp
    if (tid < (blockSize / warpSize)) {
        sum_dot      = s_dot[tid];
        sum_pred_sq  = s_pred_sq[tid];
        sum_target_sq = s_target_sq[tid];

        for (int offset = (blockSize/warpSize)/2; offset > 0; offset /= 2) {
            sum_dot      += __shfl_down_sync(mask, sum_dot, offset);
            sum_pred_sq  += __shfl_down_sync(mask, sum_pred_sq, offset);
            sum_target_sq += __shfl_down_sync(mask, sum_target_sq, offset);
        }

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

torch::Tensor coalesced_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 512;
    coalesced_cosine_similarity_loss_kernel<<<N, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &coalesced_cosine_similarity_loss_forward, "Coalesced Cosine Similarity Loss Forward (CUDA)");
}
