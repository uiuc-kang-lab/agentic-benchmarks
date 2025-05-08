#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle primitives for float values
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel: Each block processes one row and computes the cosine similarity loss
// using shared memory for intra-block reductions and warp-level shuffles for the final stage.
__global__ void sm_shfl_reduction_cosine_loss_kernel(const float* __restrict__ predictions,
                                                      const float* __restrict__ targets,
                                                      float* output,
                                                      int N,
                                                      int D) {
    // Each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Each thread processes elements with a stride of blockDim.x
    for (int i = tid; i < D; i += blockSize) {
        float p = predictions[row * D + i];
        float t = targets[row * D + i];
        sum_dot      += p * t;
        sum_pred_sq  += p * p;
        sum_target_sq+= t * t;
    }

    // First stage reduction: Intra-warp reduction using __shfl_down_sync
    sum_dot       = warp_reduce_sum(sum_dot);
    sum_pred_sq   = warp_reduce_sum(sum_pred_sq);
    sum_target_sq = warp_reduce_sum(sum_target_sq);

    // Allocate shared memory for storing warp-level partial sums
    __shared__ float s_dot[32];
    __shared__ float s_pred_sq[32];
    __shared__ float s_target_sq[32];

    int warpid = tid / warpSize;
    int lane   = tid % warpSize;

    // Lane 0 of each warp writes its warp-reduced result to shared memory
    if (lane == 0) {
        s_dot[warpid] = sum_dot;
        s_pred_sq[warpid] = sum_pred_sq;
        s_target_sq[warpid] = sum_target_sq;
    }
    __syncthreads();

    // Second stage: Final reduction by the first warp
    int numWarps = blockSize / warpSize;
    if (tid < numWarps) {
        float dot_val      = s_dot[tid];
        float pred_sq_val  = s_pred_sq[tid];
        float target_sq_val= s_target_sq[tid];

        // Reduce the partial sums within the first warp
        dot_val = warp_reduce_sum(dot_val);
        pred_sq_val = warp_reduce_sum(pred_sq_val);
        target_sq_val = warp_reduce_sum(target_sq_val);

        // Only thread 0 computes the cosine similarity loss for this row
        if (tid == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(pred_sq_val);
            float norm_target = sqrtf(target_sq_val);
            float denom = norm_pred * norm_target;
            denom = fmaxf(denom, eps);
            float cos_sim = dot_val / denom;
            // Accumulate the per-row loss scaled by 1/N
            atomicAdd(output, (1.0f - cos_sim) / static_cast<float>(N));
        }
    }
}

// Host function binding that launches the kernel
torch::Tensor sm_shfl_reduction_cosine_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 512; // Optimal block size for high occupancy
    sm_shfl_reduction_cosine_loss_kernel<<<N, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sm_shfl_reduction_cosine_loss_forward, "SM and SHFL Reduction Cosine Similarity Loss Forward (CUDA)");
}
