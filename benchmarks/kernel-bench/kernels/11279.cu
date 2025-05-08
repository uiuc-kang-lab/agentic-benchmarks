#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function to perform warp-level reduction
__device__ inline float warpReduceSum(float val) {
    // Use full warp mask
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Modular device function to compute partial sums for a given row
__device__ void compute_row_sums(const float* __restrict__ pred_row,
                                  const float* __restrict__ target_row,
                                  int D,
                                  int tid,
                                  int stride,
                                  float &sum_dot,
                                  float &sum_pred_sq,
                                  float &sum_target_sq) {
    sum_dot = 0.0f;
    sum_pred_sq = 0.0f;
    sum_target_sq = 0.0f;
    for (int i = tid; i < D; i += stride) {
        float p = pred_row[i];
        float t = target_row[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }
}

// Kernel that computes the cosine similarity loss for each row
__global__ void modular_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                         const float* __restrict__ targets,
                                                         float* output,
                                                         int N,
                                                         int D) {
    // Each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float sum_dot, sum_pred_sq, sum_target_sq;
    // Each thread computes partial sums over the row
    compute_row_sums(pred_row, target_row, D, tid, blockSize, sum_dot, sum_pred_sq, sum_target_sq);

    // Intra-warp reduction using register shuffles
    sum_dot = warpReduceSum(sum_dot);
    sum_pred_sq = warpReduceSum(sum_pred_sq);
    sum_target_sq = warpReduceSum(sum_target_sq);

    // Allocate shared memory for storing warp-level results
    extern __shared__ float shared[]; // Size is 3 * (blockDim.x / warpSize) floats
    int numWarps = blockDim.x / warpSize;
    int warpId = tid / warpSize;
    int lane = tid % warpSize;

    float* s_dot = shared;
    float* s_pred_sq = shared + numWarps;
    float* s_target_sq = shared + 2 * numWarps;

    if (lane == 0) {
        s_dot[warpId] = sum_dot;
        s_pred_sq[warpId] = sum_pred_sq;
        s_target_sq[warpId] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction across warps performed by the first warp only
    if (tid < warpSize) {
        float final_dot = (tid < numWarps) ? s_dot[tid] : 0.0f;
        float final_pred_sq = (tid < numWarps) ? s_pred_sq[tid] : 0.0f;
        float final_target_sq = (tid < numWarps) ? s_target_sq[tid] : 0.0f;

        final_dot = warpReduceSum(final_dot);
        final_pred_sq = warpReduceSum(final_pred_sq);
        final_target_sq = warpReduceSum(final_target_sq);

        if (tid == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(final_pred_sq);
            float norm_target = sqrtf(final_target_sq);
            float denom = norm_pred * norm_target;
            denom = fmaxf(denom, eps);
            float cos_sim = final_dot / denom;
            float loss = 1.0f - cos_sim;
            atomicAdd(output, loss);
        }
    }
}

// Host function to launch the modular kernel
torch::Tensor modular_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    const int block_size = 256;
    int numWarps = block_size / 32;
    size_t shared_mem = 3 * numWarps * sizeof(float);

    modular_cosine_similarity_loss_kernel<<<N, block_size, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    // Average the loss by the number of rows
    output.div_(N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &modular_cosine_similarity_loss_forward, "Modular Cosine Similarity Loss Forward (CUDA)");
}
