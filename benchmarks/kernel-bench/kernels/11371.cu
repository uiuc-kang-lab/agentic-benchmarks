#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel that computes cosine similarity loss per row using configurable block sizes
// and shared memory to load row data and perform warp-level reduction.
__global__ void cosine_similarity_loss_kernel_optimal(const float* __restrict__ predictions,
                                                        const float* __restrict__ targets,
                                                        float* output,
                                                        int N,
                                                        int D) {
    // External shared memory layout:
    // [ shared_preds (D floats), shared_targets (D floats), warp_smem (3 * num_warps floats) ]
    extern __shared__ float sdata[];
    int blockSize = blockDim.x;
    int num_warps = (blockSize + 31) / 32;

    // Partition shared memory
    float* shared_preds   = sdata;         // first D floats
    float* shared_targets = sdata + D;       // next D floats
    float* warp_smem      = sdata + 2 * D;     // remaining 3*num_warps floats

    int row = blockIdx.x;  // each block processes one row
    int tid = threadIdx.x;

    // Load the row data into shared memory (loop if D > blockDim.x)
    for (int i = tid; i < D; i += blockSize) {
        shared_preds[i] = predictions[row * D + i];
        shared_targets[i] = targets[row * D + i];
    }
    __syncthreads();

    // Each thread computes partial sums for dot product and squared norms
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

    // Reduce within each warp using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_dot += __shfl_down_sync(mask, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(mask, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(mask, sum_target_sq, offset);
    }

    int lane = tid & 31;    // lane index within warp
    int warp_id = tid / 32;
    if (lane == 0) {
        warp_smem[warp_id] = sum_dot;
        warp_smem[warp_id + num_warps] = sum_pred_sq;
        warp_smem[warp_id + 2 * num_warps] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction by the first warp
    if (tid < num_warps) {
        float dot = warp_smem[tid];
        float pred_sq = warp_smem[tid + num_warps];
        float target_sq = warp_smem[tid + 2 * num_warps];
        for (int offset = num_warps / 2; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
            pred_sq += __shfl_down_sync(0xffffffff, pred_sq, offset);
            target_sq += __shfl_down_sync(0xffffffff, target_sq, offset);
        }
        if (tid == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(pred_sq);
            float norm_target = sqrtf(target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = dot / denominator;
            atomicAdd(output, 1.0f - cos_sim);
        }
    }
}

// Host function: accepts an optional block_size parameter (allowed: 32, 64, 128, 256, 512)
// to facilitate experimentation with optimal configurations on the NVIDIA H100 GPU.

torch::Tensor cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets, int block_size = 256) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "predictions and targets must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    // Validate block_size; if invalid, default to 256
    if (block_size != 32 && block_size != 64 && block_size != 128 && block_size != 256 && block_size != 512) {
        block_size = 256;
    }

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());

    int num_warps = (block_size + 31) / 32;
    // Total shared memory: 2*D for row data + 3*num_warps for warp-level reductions
    size_t shared_mem = (2 * D + 3 * num_warps) * sizeof(float);

    // Launch one block per sample
    cosine_similarity_loss_kernel_optimal<<<N, block_size, shared_mem>>>(
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
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward with configurable block size (CUDA)",
          py::arg("predictions"), py::arg("targets"), py::arg("block_size") = 256);
}
