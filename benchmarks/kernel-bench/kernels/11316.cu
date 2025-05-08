#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                              const float* __restrict__ targets,
                                              float* output,
                                              int N,
                                              int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    int num_warps = blockDim.x / warpSize;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    for (int i = tid; i < D; i += blockDim.x) {
        float p = predictions[row * D + i];
        float t = targets[row * D + i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    sum_dot = warp_reduce_sum(sum_dot);
    sum_pred_sq = warp_reduce_sum(sum_pred_sq);
    sum_target_sq = warp_reduce_sum(sum_target_sq);

    __shared__ float shared_results[3 * 32];
    if (lane == 0) {
        shared_results[warp_id] = sum_dot;
        shared_results[num_warps + warp_id] = sum_pred_sq;
        shared_results[2 * num_warps + warp_id] = sum_target_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        if (tid < num_warps) {
            sum_dot = shared_results[lane];
            sum_pred_sq = shared_results[num_warps + lane];
            sum_target_sq = shared_results[2 * num_warps + lane];
        } else {
            sum_dot = 0.0f;
            sum_pred_sq = 0.0f;
            sum_target_sq = 0.0f;
        }

        sum_dot = warp_reduce_sum(sum_dot);
        sum_pred_sq = warp_reduce_sum(sum_pred_sq);
        sum_target_sq = warp_reduce_sum(sum_target_sq);

        if (lane == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(sum_pred_sq);
            float norm_target = sqrtf(sum_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);

            float cos_sim = sum_dot / denominator;
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

    cosine_similarity_loss_kernel<<<N, block_size>>>(
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