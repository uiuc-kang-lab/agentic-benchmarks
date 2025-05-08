#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel using stride loops with manual loop unrolling to handle workloads larger than the available threads
// The unroll factor is set to 4, and boundary conditions are verified to ensure correct accumulation.

__global__ void cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                const float* __restrict__ targets,
                                                float* output,
                                                int N,
                                                int D) {
    // Each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    const int warpSize = 32;

    // Unroll factor
    const int unroll = 4;
    int stride = blockSize * unroll; // stride for the unrolled loop

    // Pointers to the current row for predictions and targets
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Each thread starts at an index that is a multiple of the unroll factor
    int idx = tid * unroll;
    
    // Process as many groups of 'unroll' contiguous elements as possible using vector loads
    for (; idx <= D - unroll; idx += stride) {
        float4 pred_vec = *((float4*)(&pred_row[idx]));
        float4 target_vec = *((float4*)(&target_row[idx]));
        
        // Unpack vector components
        float4 p = pred_vec;
        float4 t = target_vec;
        
        // Compute dot products and squares using vector components
        sum_dot += p.x * t.x + p.y * t.y + p.z * t.z + p.w * t.w;
        sum_pred_sq += p.x * p.x + p.y * p.y + p.z * p.z + p.w * p.w;
        sum_target_sq += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
    }

    // Process any remaining elements one-by-one
    for (; idx < D; idx += 1) {
        float p = pred_row[idx];
        float t = target_row[idx];
        sum_dot       += p * t;
        sum_pred_sq   += p * p;
        sum_target_sq += t * t;
    }

    // Intra-warp reduction using shuffle built-ins
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_dot       += __shfl_down_sync(mask, sum_dot, offset);
        sum_pred_sq   += __shfl_down_sync(mask, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(mask, sum_target_sq, offset);
    }

    // Allocate shared memory for partial sums from each warp
    extern __shared__ float s[];  // Layout: [0, nWarps): dot, [nWarps, 2*nWarps): pred_sq, [2*nWarps, 3*nWarps): target_sq
    int nWarps = (blockSize + warpSize - 1) / warpSize;
    int lane = tid & (warpSize - 1);
    int warp_id = tid / warpSize;
    if(lane == 0) {
        s[warp_id]           = sum_dot;
        s[warp_id + nWarps]    = sum_pred_sq;
        s[warp_id + 2*nWarps]  = sum_target_sq;
    }
    __syncthreads();

    // Final reduction by the first warp
    if(tid < warpSize) {
        float final_dot = (tid < nWarps) ? s[tid] : 0.0f;
        float final_pred_sq = (tid < nWarps) ? s[tid + nWarps] : 0.0f;
        float final_target_sq = (tid < nWarps) ? s[tid + 2*nWarps] : 0.0f;

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            final_dot       += __shfl_down_sync(mask, final_dot, offset);
            final_pred_sq   += __shfl_down_sync(mask, final_pred_sq, offset);
            final_target_sq += __shfl_down_sync(mask, final_target_sq, offset);
        }

        if(tid == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(final_pred_sq);
            float norm_target = sqrtf(final_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = final_dot / denominator;
            atomicAdd(output, 1.0f - cos_sim);
        }
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
    int nWarps = (block_size + warpSize - 1) / warpSize;
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
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward (CUDA with stride loop and unrolling)");
}
