#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized CUDA kernel using warp-level reduction with __shfl_down_sync
__global__ void cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                 const float* __restrict__ targets,
                                                 float* output,
                                                 int N,
                                                 int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each thread works on one row of the input tensor
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Accumulate partial sums over D with a stride of blockDim.x
    for (int i = tid; i < D; i += blockDim.x) {
        float p = pred_row[i];
        float t = target_row[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Use warp-level reduction to minimize divergence.
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_dot       += __shfl_down_sync(mask, sum_dot, offset);
        sum_pred_sq   += __shfl_down_sync(mask, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(mask, sum_target_sq, offset);
    }

    // Calculate warp and lane indices
    int laneId = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x / warpSize;

    // Use shared memory to combine results from different warps
    extern __shared__ float shared[];
    // Compute number of warps in this block
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    
    // Partition shared memory into three arrays
    float* warp_dot      = shared;                      // size: numWarps
    float* warp_pred_sq  = shared + numWarps;             // size: numWarps
    float* warp_target_sq = shared + 2 * numWarps;          // size: numWarps

    // Each warp's first thread writes its reduced value
    if (laneId == 0) {
        warp_dot[warpId]      = sum_dot;
        warp_pred_sq[warpId]  = sum_pred_sq;
        warp_target_sq[warpId]= sum_target_sq;
    }
    __syncthreads();

    // Thread 0 aggregates results from all warps in the block
    if (tid == 0) {
        float block_dot = 0.0f;
        float block_pred_sq = 0.0f;
        float block_target_sq = 0.0f;
        for (int i = 0; i < numWarps; i++) {
            block_dot       += warp_dot[i];
            block_pred_sq   += warp_pred_sq[i];
            block_target_sq += warp_target_sq[i];
        }
        
        // Compute cosine similarity loss with numerical stability
        const float eps = 1e-8f;
        float norm_pred = sqrtf(block_pred_sq);
        float norm_target = sqrtf(block_target_sq);
        float denominator = norm_pred * norm_target;
        denominator = fmaxf(denominator, eps);
        float cos_sim = block_dot / denominator;

        // Atomic addition to accumulate the loss from each row
        atomicAdd(output, 1.0f - cos_sim);
    }
}

// Forward function called from PyTorch
torch::Tensor cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    const int block_size = 256; // Ensure block size is a multiple of warp size (32)
    int numWarps = (block_size + 31) / 32;
    // Shared memory: three arrays of size numWarps each
    size_t shared_mem = numWarps * 3 * sizeof(float);

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
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward (CUDA optimized with warp shuffle)");
}
