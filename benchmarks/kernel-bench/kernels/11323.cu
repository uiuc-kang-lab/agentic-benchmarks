#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                               const float* __restrict__ targets,
                                               float* output,
                                               int N,
                                               int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    
    const int warpSize = 32;
    int lane = tid & (warpSize - 1);
    int warp_id = tid / warpSize;

    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    // Initialize partial sums
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Manual unroll for the main computation loop when D is large
    #pragma unroll 4
    for (int i = tid; i < D; i += blockSize) {
        float p = pred_row[i];
        float t = target_row[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Unrolled warp reduction using shuffle instructions
    unsigned int mask = 0xffffffff;
    sum_dot += __shfl_down_sync(mask, sum_dot, 16);
    sum_pred_sq += __shfl_down_sync(mask, sum_pred_sq, 16);
    sum_target_sq += __shfl_down_sync(mask, sum_target_sq, 16);

    sum_dot += __shfl_down_sync(mask, sum_dot, 8);
    sum_pred_sq += __shfl_down_sync(mask, sum_pred_sq, 8);
    sum_target_sq += __shfl_down_sync(mask, sum_target_sq, 8);

    sum_dot += __shfl_down_sync(mask, sum_dot, 4);
    sum_pred_sq += __shfl_down_sync(mask, sum_pred_sq, 4);
    sum_target_sq += __shfl_down_sync(mask, sum_target_sq, 4);

    sum_dot += __shfl_down_sync(mask, sum_dot, 2);
    sum_pred_sq += __shfl_down_sync(mask, sum_pred_sq, 2);
    sum_target_sq += __shfl_down_sync(mask, sum_target_sq, 2);

    sum_dot += __shfl_down_sync(mask, sum_dot, 1);
    sum_pred_sq += __shfl_down_sync(mask, sum_pred_sq, 1);
    sum_target_sq += __shfl_down_sync(mask, sum_target_sq, 1);

    // Use shared memory to store per-warp reductions
    extern __shared__ float shared[];
    int nWarps = (blockSize + warpSize - 1) / warpSize;
    
    if (lane == 0) {
        shared[warp_id] = sum_dot;
        shared[warp_id + nWarps] = sum_pred_sq;
        shared[warp_id + 2 * nWarps] = sum_target_sq;
    }
    __syncthreads();

    // Unrolled final reduction for first warp
    if (warp_id == 0 && lane < nWarps) {
        float final_dot = shared[lane];
        float final_pred_sq = shared[lane + nWarps];
        float final_target_sq = shared[lane + 2 * nWarps];

        // Unrolled warp reduction for final values
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            final_dot += __shfl_down_sync(mask, final_dot, offset);
            final_pred_sq += __shfl_down_sync(mask, final_pred_sq, offset);
            final_target_sq += __shfl_down_sync(mask, final_target_sq, offset);
        }

        if (lane == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(final_pred_sq);
            float norm_target = sqrtf(final_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = final_dot / denominator;
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
    int nWarps = (block_size + 31) / 32;
    size_t shared_mem = 3 * nWarps * sizeof(float);

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