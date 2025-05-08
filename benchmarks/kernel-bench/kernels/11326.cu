#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function for warp-level reduction
__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Device function to compute partial sums
__device__ void compute_partial_sums(const float* pred_row, const float* target_row, int D, int tid, int blockSize, float &sum_dot, float &sum_pred_sq, float &sum_target_sq) {
    for (int i = tid; i < D; i += blockSize) {
        float p = pred_row[i];
        float t = target_row[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }
}

// Device function to perform final reduction and update output
__device__ void finalize_reduction(float* shared, int tid, int nWarps, float &final_dot, float &final_pred_sq, float &final_target_sq, float* output) {
    // Parallel reduction for the final sums
    if (tid < nWarps) {
        __syncwarp();
        float dot = final_dot;
        float pred_sq = final_pred_sq;
        float target_sq = final_target_sq;
        
        #pragma unroll
        for (int stride = nWarps/2; stride > 0; stride >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, stride);
            pred_sq += __shfl_down_sync(0xffffffff, pred_sq, stride);
            target_sq += __shfl_down_sync(0xffffffff, target_sq, stride);
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

// Main CUDA kernel
__global__ void cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                               const float* __restrict__ targets,
                                               float* output,
                                               int N,
                                               int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warpSize = 32;
    int lane = tid & (warpSize - 1);
    int warp_id = tid / warpSize;

    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    compute_partial_sums(pred_row, target_row, D, tid, blockSize, sum_dot, sum_pred_sq, sum_target_sq);

    sum_dot = warp_reduce_sum(sum_dot);
    sum_pred_sq = warp_reduce_sum(sum_pred_sq);
    sum_target_sq = warp_reduce_sum(sum_target_sq);

    extern __shared__ float shared[];
    int nWarps = (blockSize + warpSize - 1) / warpSize;
    if (lane == 0) {
        shared[warp_id] = sum_dot;
        shared[warp_id + nWarps] = sum_pred_sq;
        shared[warp_id + 2 * nWarps] = sum_target_sq;
    }
    __syncthreads();

    float final_dot = 0.0f;
    float final_pred_sq = 0.0f;
    float final_target_sq = 0.0f;
    if (tid < nWarps) {
        final_dot = shared[tid];
        final_pred_sq = shared[tid + nWarps];
        final_target_sq = shared[tid + 2 * nWarps];
    }
    __syncthreads();

    finalize_reduction(shared, tid, nWarps, final_dot, final_pred_sq, final_target_sq, output);
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