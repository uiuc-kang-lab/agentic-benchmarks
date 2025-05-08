#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This CUDA kernel uses warp-level primitives to reduce shared memory traffic for reductions
__global__ void cosine_similarity_loss_kernel_warp(const float* __restrict__ predictions,
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

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Loop over elements with stride equal to blockDim.x
    for (int i = tid; i < D; i += blockSize) {
        float p = pred_row[i];
        float t = target_row[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    // Assume warp size of 32
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_dot += __shfl_down_sync(mask, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(mask, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(mask, sum_target_sq, offset);
    }

    // Identify lane and warp ID
    int lane = tid & 31;        // tid % 32
    int warpId = tid >> 5;        // tid / 32
    int numWarps = (blockSize + warpSize - 1) / warpSize;

    // Allocate minimal shared memory for storing warp sums
    extern __shared__ float shared[];  // size: 3 * numWarps floats
    float* s_dot      = shared;
    float* s_pred_sq  = s_dot + numWarps;
    float* s_target_sq= s_pred_sq + numWarps;

    // Each warp's lane 0 writes its reduced sum to shared memory
    if (lane == 0) {
        s_dot[warpId] = sum_dot;
        s_pred_sq[warpId] = sum_pred_sq;
        s_target_sq[warpId] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction: only threads in the first warp participate
    if (tid < numWarps) {
        sum_dot      = s_dot[tid];
        sum_pred_sq  = s_pred_sq[tid];
        sum_target_sq= s_target_sq[tid];
        
        // Use warp-level reduction if numWarps > 1
        for (int offset = (numWarps >> 1); offset > 0; offset /= 2) {
            sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
            sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
            sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
        }

        // Thread 0 in the first warp computes the final loss
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
    int numWarps = (block_size + 31) / 32;
    size_t shared_mem = 3 * numWarps * sizeof(float); // only needed for warp sums

    // Launch one block per sample
    cosine_similarity_loss_kernel_warp<<<N, block_size, shared_mem>>>(
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
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward with warp-level primitives (CUDA)");
}
