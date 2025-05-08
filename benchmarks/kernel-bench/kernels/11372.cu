#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <algorithm>

// Kernel that processes a chunk of rows. Each block processes one row.
// It uses a contiguous shared-memory buffer to load the row data (predictions and targets) and then
// performs a two-stage reduction (intra-warp using shuffle and inter-warp using shared memory) to compute
// the cosine similarity loss for that row.

__global__ void cosine_similarity_loss_kernel_stream(const float* __restrict__ predictions,
                                                       const float* __restrict__ targets,
                                                       float* output,
                                                       int offset,      // not used when pointers are pre-offset
                                                       int num_rows,
                                                       int D) {
    // Each block processes one row in the chunk. Global row index = offset + blockIdx.x
    int row = blockIdx.x;
    if (row >= num_rows) return;  
    int global_row = offset + row;  // For clarity; if pointers are pre-offset, offset=0.

    // Use one contiguous shared memory allocation for two purposes:
    // [0, D)       -> shared_preds (row of predictions)
    // [D, 2*D)     -> shared_targets (row of targets)
    // [2*D, 2*D + 3*numWarps) -> warp-level reduction arrays: dot, pred_sq, target_sq
    extern __shared__ float s[];
    float* shared_preds   = s;
    float* shared_targets = s + D;
    
    int numWarps = (blockDim.x + 31) / 32;
    float* s_dot      = s + 2 * D;
    float* s_pred_sq  = s_dot + numWarps;
    float* s_target_sq= s_pred_sq + numWarps;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Load the row from global memory into shared memory. Only the first D threads do this.
    if (tid < D) {
        shared_preds[tid] = predictions[global_row * D + tid];
        shared_targets[tid] = targets[global_row * D + tid];
    }
    __syncthreads();

    // Each thread computes partial sums over the row using a strided loop.
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

    // Intra-warp reduction using shuffle operations
    unsigned int mask = 0xffffffff;
    for (int offset_sh = 16; offset_sh > 0; offset_sh /= 2) {
        sum_dot += __shfl_down_sync(mask, sum_dot, offset_sh);
        sum_pred_sq += __shfl_down_sync(mask, sum_pred_sq, offset_sh);
        sum_target_sq += __shfl_down_sync(mask, sum_target_sq, offset_sh);
    }

    int lane = tid & 31;   // lane index within the warp
    int warpId = tid >> 5; // warp index within the block
    if (lane == 0) {
        s_dot[warpId] = sum_dot;
        s_pred_sq[warpId] = sum_pred_sq;
        s_target_sq[warpId] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction: let the first warp accumulate the values from all warps
    if (tid < numWarps) {
        sum_dot = s_dot[tid];
        sum_pred_sq = s_pred_sq[tid];
        sum_target_sq = s_target_sq[tid];
        for (int offset_sh = (numWarps >> 1); offset_sh > 0; offset_sh /= 2) {
            sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset_sh);
            sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset_sh);
            sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset_sh);
        }
        if (tid == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(sum_pred_sq);
            float norm_target = sqrtf(sum_target_sq);
            float denom = norm_pred * norm_target;
            denom = fmaxf(denom, eps);
            float cos_sim = sum_dot / denom;
            // Atomic accumulation across rows
            atomicAdd(output, 1.0f - cos_sim);
        }
    }
}

// Host function that divides the batch into chunks and launches asynchronous kernel calls on multiple CUDA streams.
// This overlaps computation with memory transfer (if present) and pipelines kernel execution for improved throughput.

torch::Tensor cosine_similarity_loss_forward_stream(torch::Tensor predictions, torch::Tensor targets) {
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
    // Shared memory: 2*D for the row data + 3*numWarps for warp-reduction arrays
    size_t shared_mem = (2 * D + 3 * numWarps) * sizeof(float);

    // Decide on number of streams; using 4 streams if there are enough rows
    int nstreams = (N < 4) ? 1 : 4;
    int chunk_size = (N + nstreams - 1) / nstreams;

    std::vector<cudaStream_t> streams(nstreams);
    for (int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch kernels on each stream for its chunk of rows
    for (int i = 0; i < nstreams; i++) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, N);
        int chunk_rows = end - start;
        if (chunk_rows <= 0) continue;

        // Offset the predictions and targets pointers by start * D
        const float* pred_ptr = predictions.data_ptr<float>() + start * D;
        const float* targ_ptr = targets.data_ptr<float>() + start * D;

        // Launch kernel with one block per row in this chunk. The 'offset' parameter is set to 0
        // because the pointers are already offset to the beginning of the chunk.
        cosine_similarity_loss_kernel_stream<<<chunk_rows, block_size, shared_mem, streams[i]>>>(
            pred_ptr,
            targ_ptr,
            output.data_ptr<float>(),
            0,
            chunk_rows,
            D
        );
    }

    // Synchronize all streams
    for (int i = 0; i < nstreams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    output.div_(N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cosine_similarity_loss_forward_stream, "Cosine Similarity Loss Forward with Streaming Pipelines (CUDA)");
}
