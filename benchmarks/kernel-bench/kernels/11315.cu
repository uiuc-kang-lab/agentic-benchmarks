#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel using stride loops and warp-level reductions
__global__ void cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                               const float* __restrict__ targets,
                                               float* output,
                                               int N,
                                               int D) {
    // Each block handles one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    
    // Define warp parameters
    const int warpSize = 32;
    int lane = tid & (warpSize - 1);
    int warp_id = tid / warpSize;

    // Pointers for the current row
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    // Initialize partial sums
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Stride loop: handles cases where D > blockDim.x
    for (int i = tid; i < D; i += blockSize) {
        float p = pred_row[i];
        float t = target_row[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Intra-warp reduction using shuffle instructions
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_dot += __shfl_down_sync(mask, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(mask, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(mask, sum_target_sq, offset);
    }

    // Use shared memory to store per-warp reductions
    extern __shared__ float shared[];  // size: 3 * nWarps floats
    // Compute number of warps per block
    int nWarps = (blockSize + warpSize - 1) / warpSize;
    if (lane == 0) {
        shared[warp_id] = sum_dot;
        shared[warp_id + nWarps] = sum_pred_sq;
        shared[warp_id + 2 * nWarps] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction from warp-level results by first nWarps threads
    float final_dot = 0.0f;
    float final_pred_sq = 0.0f;
    float final_target_sq = 0.0f;
    if (tid < nWarps) {
        final_dot = shared[tid];
        final_pred_sq = shared[tid + nWarps];
        final_target_sq = shared[tid + 2 * nWarps];
    }
    __syncthreads();

    // Thread 0 performs the final summation and writes result
    if (tid == 0) {
        for (int i = 1; i < nWarps; i++) {
            final_dot += shared[i];
            final_pred_sq += shared[i + nWarps];
            final_target_sq += shared[i + 2 * nWarps];
        }
        const float eps = 1e-8f;
        float norm_pred = sqrtf(final_pred_sq);
        float norm_target = sqrtf(final_target_sq);
        float denominator = norm_pred * norm_target;
        denominator = fmaxf(denominator, eps);
        float cos_sim = final_dot / denominator;
        atomicAdd(output, 1.0f - cos_sim);
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
    const int block_size = 256;
    // Calculate number of warps per block and required shared memory
    int nWarps = (block_size + 31) / 32;
    size_t shared_mem = 3 * nWarps * sizeof(float);

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
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward (CUDA)");
}
