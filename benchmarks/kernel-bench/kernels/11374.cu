#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized kernel using shared memory for intra-block reduction and warp-level primitives for final reduction
__global__ void cosine_similarity_loss_kernel_optimized(const float* __restrict__ predictions,
                                                         const float* __restrict__ targets,
                                                         float* output,
                                                         int N,
                                                         int D) {
    // Each block processes one row (sample)
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    // Each thread accumulates partial sums for dot product and squared norms
    float dot = 0.0f;
    float norm_pred = 0.0f;
    float norm_target = 0.0f;

    for (int i = tid; i < D; i += blockSize) {
        float p = pred_row[i];
        float t = target_row[i];
        dot += p * t;
        norm_pred += p * p;
        norm_target += t * t;
    }

    // Perform warp-level reduction within each warp using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        dot        += __shfl_down_sync(mask, dot, offset);
        norm_pred  += __shfl_down_sync(mask, norm_pred, offset);
        norm_target+= __shfl_down_sync(mask, norm_target, offset);
    }

    // Each warp's lane 0 writes its partial result to shared memory
    int lane = tid & 31;
    int warpId = tid / 32;
    int numWarps = blockSize / 32;

    // Shared memory to hold warp-level results: three arrays of size 'numWarps'
    extern __shared__ float shared[];
    float* s_dot       = shared;                // size: numWarps
    float* s_norm_pred = s_dot + numWarps;        // size: numWarps
    float* s_norm_target = s_norm_pred + numWarps;  // size: numWarps

    if (lane == 0) {
        s_dot[warpId] = dot;
        s_norm_pred[warpId] = norm_pred;
        s_norm_target[warpId] = norm_target;
    }
    __syncthreads();

    // Final reduction across warp results done by the first 'numWarps' threads
    if (tid < numWarps) {
        float sum_dot = s_dot[tid];
        float sum_norm_pred = s_norm_pred[tid];
        float sum_norm_target = s_norm_target[tid];

        for (int offset = numWarps / 2; offset > 0; offset /= 2) {
            if (tid < offset) {
                sum_dot += s_dot[tid + offset];
                sum_norm_pred += s_norm_pred[tid + offset];
                sum_norm_target += s_norm_target[tid + offset];
                s_dot[tid] = sum_dot;
                s_norm_pred[tid] = sum_norm_pred;
                s_norm_target[tid] = sum_norm_target;
            }
            __syncthreads();
        }

        // Thread 0 computes the final cosine similarity loss
        if (tid == 0) {
            const float eps = 1e-8f;
            float norm_val = sqrtf(s_norm_pred[0]) * sqrtf(s_norm_target[0]);
            norm_val = fmaxf(norm_val, eps);
            float cos_sim = s_dot[0] / norm_val;
            atomicAdd(output, 1.0f - cos_sim);
        }
    }
}

// Host function to launch the kernel
torch::Tensor cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;  // Must be a multiple of 32
    TORCH_CHECK(block_size % 32 == 0, "block_size must be a multiple of 32");
    int numWarps = block_size / 32;
    // Allocate shared memory for three arrays each of size 'numWarps'
    size_t shared_mem = 3 * numWarps * sizeof(float);

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
    m.def("forward", &cosine_similarity_loss_forward, "Optimized Cosine Similarity Loss Forward (CUDA)");
}
