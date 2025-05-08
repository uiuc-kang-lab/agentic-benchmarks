#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized kernel using shared memory efficiently and minimizing syncthreads
__global__ void cosine_similarity_loss_kernel_optimized(const float* __restrict__ predictions,
                                                        const float* __restrict__ targets,
                                                        float* output,
                                                        int N,
                                                        int D) {
    extern __shared__ float shared_mem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float* shared_preds = shared_mem;
    float* shared_targets = shared_preds + D;

    // Load data into shared memory only once per row
    if (tid < D) {
        shared_preds[tid] = predictions[row * D + tid];
        shared_targets[tid] = targets[row * D + tid];
    }
    __syncthreads();

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    for (int i = tid; i < D; i += blockSize) {
        float p = shared_preds[i];
        float t = shared_targets[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_dot += __shfl_down_sync(mask, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(mask, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(mask, sum_target_sq, offset);
    }

    int lane = tid & 31;        // tid % 32
    int warpId = tid >> 5;      // tid / 32
    int numWarps = (blockSize + warpSize - 1) / warpSize;

    extern __shared__ float shared[];  // size: 3 * numWarps floats
    float* s_dot      = shared;
    float* s_pred_sq  = s_dot + numWarps;
    float* s_target_sq= s_pred_sq + numWarps;

    if (lane == 0) {
        s_dot[warpId] = sum_dot;
        s_pred_sq[warpId] = sum_pred_sq;
        s_target_sq[warpId] = sum_target_sq;
    }
    __syncthreads(); // Only synchronize here once for warp-level reduction

    if (tid < numWarps) {
        sum_dot      = s_dot[tid];
        sum_pred_sq  = s_pred_sq[tid];
        sum_target_sq= s_target_sq[tid];

        for (int offset = (numWarps >> 1); offset > 0; offset /= 2) {
            sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
            sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
            sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
        }

        if (tid == 0) {
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

// Host function that wraps the CUDA kernel
// This function performs necessary input checks and kernel launch

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
    size_t shared_mem = (2 * D + 3 * ((block_size + 31) / 32)) * sizeof(float); // memory for shared_preds, shared_targets, and warp sums

    // Launch one block per sample
    cosine_similarity_loss_kernel_optimized<<<N, block_size, shared_mem>>>(
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
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward with optimized shared memory (CUDA)");
}
