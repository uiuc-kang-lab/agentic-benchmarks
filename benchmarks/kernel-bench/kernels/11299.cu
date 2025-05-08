#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Use a template for block size so we can easily experiment with different sizes.
template<int BLOCK_SIZE>
__global__ void block_size_tuned_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                              const float* __restrict__ targets,
                                                              float* output,
                                                              const int N,
                                                              const int D) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    const int warps_per_block = BLOCK_SIZE / warpSize;
    
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    #pragma unroll 4
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float pred = predictions[row * D + i];
        float target = targets[row * D + i];
        sum_dot += pred * target;
        sum_pred_sq += pred * pred;
        sum_target_sq += target * target;
    }

    sum_dot = warp_reduce_sum(sum_dot);
    sum_pred_sq = warp_reduce_sum(sum_pred_sq);
    sum_target_sq = warp_reduce_sum(sum_target_sq);

    __shared__ float s_dot[32];
    __shared__ float s_pred_sq[32];
    __shared__ float s_target_sq[32];

    if (lane_id == 0) {
        s_dot[warp_id] = sum_dot;
        s_pred_sq[warp_id] = sum_pred_sq;
        s_target_sq[warp_id] = sum_target_sq;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < warps_per_block) {
        sum_dot = s_dot[lane_id];
        sum_pred_sq = s_pred_sq[lane_id];
        sum_target_sq = s_target_sq[lane_id];

        sum_dot = warp_reduce_sum(sum_dot);
        sum_pred_sq = warp_reduce_sum(sum_pred_sq);
        sum_target_sq = warp_reduce_sum(sum_target_sq);

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

torch::Tensor block_size_tuned_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    
    // Testing different block sizes
    if (D <= 256) {
        block_size_tuned_cosine_similarity_loss_kernel<128><<<N, 128>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D);
    } else if (D <= 512) {
        block_size_tuned_cosine_similarity_loss_kernel<256><<<N, 256>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D);
    } else if (D <= 1024) {
        block_size_tuned_cosine_similarity_loss_kernel<384><<<N, 384>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D);
    } else {
        block_size_tuned_cosine_similarity_loss_kernel<512><<<N, 512>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_size_tuned_cosine_similarity_loss_forward, "Block Size Tuned Cosine Similarity Loss Forward (CUDA)");
}