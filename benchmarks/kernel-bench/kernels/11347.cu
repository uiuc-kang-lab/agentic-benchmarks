#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel minimizes warp divergence by padding shared memory to a full warp size,
// so that the final reduction is performed uniformly for all lanes in the first warp.
// Unused entries are pre-filled with zeros to avoid divergent conditional logic.

__global__ void cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                               const float* __restrict__ targets,
                                               float* output,
                                               int N,
                                               int D) {
    // Each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warpSize = 32;
    int lane = tid & (warpSize - 1);
    int warp_id = tid / warpSize;

    // Pointers to the row data
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float sum_dot = 0.f;
    float sum_pred_sq = 0.f;
    float sum_target_sq = 0.f;

    // Process elements with a stride loop
    for (int i = tid; i < D; i += blockSize) {
        float p = pred_row[i];
        float t = target_row[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Intra-warp reduction using warp shuffle operations
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_dot       += __shfl_down_sync(mask, sum_dot, offset);
        sum_pred_sq   += __shfl_down_sync(mask, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(mask, sum_target_sq, offset);
    }

    // Allocate shared memory padded to warpSize for uniform final reduction
    // Layout: [0, warpSize): dot, [warpSize, 2*warpSize): pred_sq, [2*warpSize, 3*warpSize): target_sq
    extern __shared__ float s[];
    if (lane == 0) {
        s[warp_id] = sum_dot;
        s[warp_id + warpSize] = sum_pred_sq;
        s[warp_id + 2 * warpSize] = sum_target_sq;
    }
    __syncthreads();

    // Compute number of warps used in the block
    int nWarps = (blockSize + warpSize - 1) / warpSize;

    // Thread 0 pads the remainder of the shared memory with zeros to avoid divergence in final reduction
    if (tid == 0) {
        for (int i = nWarps; i < warpSize; i++) {
            s[i] = 0.f;
            s[i + warpSize] = 0.f;
            s[i + 2 * warpSize] = 0.f;
        }
    }
    __syncthreads();

    // Final reduction: first warp processes a full set of warpSize elements uniformly
    float final_dot = 0.f;
    float final_pred_sq = 0.f;
    float final_target_sq = 0.f;
    if (tid < warpSize) {
        final_dot = s[tid];
        final_pred_sq = s[tid + warpSize];
        final_target_sq = s[tid + 2 * warpSize];

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            final_dot       += __shfl_down_sync(mask, final_dot, offset);
            final_pred_sq   += __shfl_down_sync(mask, final_pred_sq, offset);
            final_target_sq += __shfl_down_sync(mask, final_target_sq, offset);
        }
    }

    if (tid == 0) {
        const float eps = 1e-8f;
        float norm_pred = sqrtf(final_pred_sq);
        float norm_target = sqrtf(final_target_sq);
        float denominator = norm_pred * norm_target;
        denominator = fmaxf(denominator, eps);
        float cos_sim = final_dot / denominator;
        atomicAdd(output, 1.0f - cos_sim);
    }
}

// Host function to launch the CUDA kernel
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
    const int warpSize = 32;
    // Shared memory allocated for 3 arrays padded to warpSize
    size_t shared_mem = 3 * warpSize * sizeof(float);

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
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward (CUDA uniform padded warp reduction)");
}
