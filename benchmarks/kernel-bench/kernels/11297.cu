#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void block_tuned_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                          const float* __restrict__ targets,
                                                          float* output,
                                                          const int N,
                                                          const int D) {
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;
    
    // Calculate number of elements per thread
    const int items_per_thread = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Process elements with stride equal to block size
    #pragma unroll 4
    for (int i = 0; i < items_per_thread; i++) {
        const int idx = tid + i * BLOCK_SIZE;
        if (idx < D) {
            const float pred = pred_row[idx];
            const float target = target_row[idx];
            sum_dot += pred * target;
            sum_pred_sq += pred * pred;
            sum_target_sq += target * target;
        }
    }

    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
    }

    __shared__ float s_dot[WARPS_PER_BLOCK];
    __shared__ float s_pred_sq[WARPS_PER_BLOCK];
    __shared__ float s_target_sq[WARPS_PER_BLOCK];

    // Store warp results to shared memory
    if (lane_id == 0) {
        s_dot[warp_id] = sum_dot;
        s_pred_sq[warp_id] = sum_pred_sq;
        s_target_sq[warp_id] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0 && lane_id < WARPS_PER_BLOCK) {
        sum_dot = s_dot[lane_id];
        sum_pred_sq = s_pred_sq[lane_id];
        sum_target_sq = s_target_sq[lane_id];

        // Warp-level reduction for final results
        #pragma unroll
        for (int offset = WARPS_PER_BLOCK/2; offset > 0; offset /= 2) {
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

torch::Tensor block_tuned_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);
    auto output = torch::zeros({1}, predictions.options());

    // Choose block size based on D dimension
    if (D <= 256) {
        block_tuned_cosine_similarity_loss_kernel<128><<<N, 128>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D);
    } else if (D <= 512) {
        block_tuned_cosine_similarity_loss_kernel<256><<<N, 256>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D);
    } else if (D <= 1024) {
        block_tuned_cosine_similarity_loss_kernel<384><<<N, 384>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D);
    } else {
        block_tuned_cosine_similarity_loss_kernel<512><<<N, 512>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_tuned_cosine_similarity_loss_forward, "Block Tuned Cosine Similarity Loss Forward (CUDA)");
}