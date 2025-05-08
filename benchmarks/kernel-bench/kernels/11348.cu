#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel using vectorized loads, warp-level reductions, and manual loop unrolling
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

    // Pointers for the current row
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Use vectorized loads with float4 when possible
    const int vec_elements = 4;
    const int D_aligned = (D / vec_elements) * vec_elements;
    const float4* pred_row_vec = reinterpret_cast<const float4*>(pred_row);
    const float4* target_row_vec = reinterpret_cast<const float4*>(target_row);

    // Process aligned elements; unroll the loop to reduce overhead
    #pragma unroll 4
    for (int i = tid * vec_elements; i < D_aligned; i += blockSize * vec_elements) {
        float4 p_vec = __ldg(&pred_row_vec[i / vec_elements]);
        float4 t_vec = __ldg(&target_row_vec[i / vec_elements]);
        sum_dot += p_vec.x * t_vec.x + p_vec.y * t_vec.y + p_vec.z * t_vec.z + p_vec.w * t_vec.w;
        sum_pred_sq += p_vec.x * p_vec.x + p_vec.y * p_vec.y + p_vec.z * p_vec.z + p_vec.w * p_vec.w;
        sum_target_sq += t_vec.x * t_vec.x + t_vec.y * t_vec.y + t_vec.z * t_vec.z + t_vec.w * t_vec.w;
    }

    // Process remaining elements
    #pragma unroll 4
    for (int i = D_aligned + tid; i < D; i += blockSize) {
        float p = __ldg(&pred_row[i]);
        float t = __ldg(&target_row[i]);
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Intra-warp reduction using warp shuffle with unrolled loop
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_dot       += __shfl_down_sync(mask, sum_dot, offset);
        sum_pred_sq   += __shfl_down_sync(mask, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(mask, sum_target_sq, offset);
    }

    // Use shared memory to store per-warp results
    extern __shared__ float shared[];  // layout: [0, nWarps): dot, [nWarps, 2*nWarps): pred_sq, [2*nWarps, 3*nWarps): target_sq
    int nWarps = (blockSize + warpSize - 1) / warpSize;
    if (lane == 0) {
        shared[warp_id] = sum_dot;
        shared[warp_id + nWarps] = sum_pred_sq;
        shared[warp_id + 2 * nWarps] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction by the first warp, unroll the reduction loop
    float final_dot = 0.0f;
    float final_pred_sq = 0.0f;
    float final_target_sq = 0.0f;
    if (tid < warpSize) {
        final_dot = (tid < nWarps) ? shared[tid] : 0.0f;
        final_pred_sq = (tid < nWarps) ? shared[tid + nWarps] : 0.0f;
        final_target_sq = (tid < nWarps) ? shared[tid + 2 * nWarps] : 0.0f;
        #pragma unroll
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
    const int warpSizeHost = 32;
    int nWarps = (block_size + warpSizeHost - 1) / warpSizeHost;
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
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward (CUDA with unrolled loops)");
}
