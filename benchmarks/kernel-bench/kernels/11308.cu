#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Templated kernel that uses different block sizes for tuning performance
template <int BLOCK_SIZE>
__global__ void blocksize_tuning_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                                 const float* __restrict__ targets,
                                                                 float* output,
                                                                 const int N,
                                                                 const int D) {
    // Each block processes one row
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Iterate over the D dimension in strides of BLOCK_SIZE
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float p = predictions[row * D + i];
        float t = targets[row * D + i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Warp-level reduction using shuffle within each warp (warp size is 32)
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
    }

    // Allocate shared memory for partial results from each warp
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ float s_dot[NUM_WARPS];
    __shared__ float s_pred_sq[NUM_WARPS];
    __shared__ float s_target_sq[NUM_WARPS];
    
    int warp_id = tid / 32;
    int lane = tid & 31;  // tid % 32
    if (lane == 0) {
        s_dot[warp_id] = sum_dot;
        s_pred_sq[warp_id] = sum_pred_sq;
        s_target_sq[warp_id] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction: first warp reduces the partial sums
    float final_dot = 0.0f;
    float final_pred_sq = 0.0f;
    float final_target_sq = 0.0f;
    if (tid < NUM_WARPS) {
        final_dot = s_dot[tid];
        final_pred_sq = s_pred_sq[tid];
        final_target_sq = s_target_sq[tid];
        
        // Reduce within the first warp
        for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
            final_dot += __shfl_down_sync(0xffffffff, final_dot, offset);
            final_pred_sq += __shfl_down_sync(0xffffffff, final_pred_sq, offset);
            final_target_sq += __shfl_down_sync(0xffffffff, final_target_sq, offset);
        }
        
        if (tid == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(final_pred_sq);
            float norm_target = sqrtf(final_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = final_dot / denominator;
            // Accumulate loss over rows and average by dividing by N
            atomicAdd(output, (1.0f - cos_sim) / N);
        }
    }
}

// Host binding function with block size dispatching
torch::Tensor blocksize_tuning_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);
    auto output = torch::zeros({1}, predictions.options());

    // Experiment with a range of block sizes based on the D dimension
    if (D <= 64) {
        blocksize_tuning_cosine_similarity_loss_kernel<32><<<N, 32>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D);
    } else if (D <= 128) {
        blocksize_tuning_cosine_similarity_loss_kernel<64><<<N, 64>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D);
    } else if (D <= 256) {
        blocksize_tuning_cosine_similarity_loss_kernel<128><<<N, 128>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D);
    } else if (D <= 512) {
        blocksize_tuning_cosine_similarity_loss_kernel<256><<<N, 256>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D);
    } else {
        blocksize_tuning_cosine_similarity_loss_kernel<512><<<N, 512>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &blocksize_tuning_cosine_similarity_loss_forward, "Blocksize Tuning Cosine Similarity Loss Forward (CUDA)");
}
