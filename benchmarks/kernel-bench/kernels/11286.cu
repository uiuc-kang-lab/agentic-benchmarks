#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void hybrid_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                      const float* __restrict__ targets,
                                                      float* output,
                                                      int N,
                                                      int D) {
    extern __shared__ float s_data[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    for (int i = tid; i < D; i += stride) {
        float p = pred_row[i];
        float t = target_row[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    sum_dot = warpReduceSum(sum_dot);
    sum_pred_sq = warpReduceSum(sum_pred_sq);
    sum_target_sq = warpReduceSum(sum_target_sq);

    int lane = tid & (warpSize - 1);
    int warp_id = tid >> 5;

    if (lane == 0) {
        s_data[warp_id] = sum_dot;
        s_data[warp_id + 32] = sum_pred_sq;
        s_data[warp_id + 64] = sum_target_sq;
    }

    __syncthreads();

    if (tid < warpSize) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float final_dot = (tid < num_warps) ? s_data[tid] : 0.0f;
        float final_pred_sq = (tid < num_warps) ? s_data[tid + 32] : 0.0f;
        float final_target_sq = (tid < num_warps) ? s_data[tid + 64] : 0.0f;

        final_dot = warpReduceSum(final_dot);
        final_pred_sq = warpReduceSum(final_pred_sq);
        final_target_sq = warpReduceSum(final_target_sq);

        if (tid == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(final_pred_sq);
            float norm_target = sqrtf(final_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = final_dot / denominator;
            atomicAdd(output, (1.0f - cos_sim) / N);
        }
    }
}

torch::Tensor hybrid_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    const int block_size = 256;
    size_t shared_mem = 96 * sizeof(float); // 3 * 32 floats

    hybrid_cosine_similarity_loss_kernel<<<N, block_size, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hybrid_cosine_similarity_loss_forward, "Hybrid Cosine Similarity Loss Forward (CUDA)");
}