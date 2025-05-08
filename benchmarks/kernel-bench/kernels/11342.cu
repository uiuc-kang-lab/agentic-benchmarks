#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel using vectorized loads and warp-level reductions
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

    // Align pointers for vector loads
    const float4* pred_row_vec = reinterpret_cast<const float4*>(predictions + row * D);
    const float4* target_row_vec = reinterpret_cast<const float4*>(targets + row * D);
    
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Vector loads for aligned elements
    const int vec_elements = 4;
    const int D_aligned = D / vec_elements * vec_elements;
    
    // Process aligned elements using vector loads
    for (int i = tid * vec_elements; i < D_aligned; i += blockSize * vec_elements) {
        float4 pred_vec = __ldg(&pred_row_vec[i/vec_elements]);
        float4 target_vec = __ldg(&target_row_vec[i/vec_elements]);
        
        sum_dot += pred_vec.x * target_vec.x + 
                  pred_vec.y * target_vec.y + 
                  pred_vec.z * target_vec.z + 
                  pred_vec.w * target_vec.w;
                  
        sum_pred_sq += pred_vec.x * pred_vec.x + 
                      pred_vec.y * pred_vec.y + 
                      pred_vec.z * pred_vec.z + 
                      pred_vec.w * pred_vec.w;
                      
        sum_target_sq += target_vec.x * target_vec.x + 
                        target_vec.y * target_vec.y + 
                        target_vec.z * target_vec.z + 
                        target_vec.w * target_vec.w;
    }

    // Handle remaining elements
    for (int i = D_aligned + tid; i < D; i += blockSize) {
        float pred = __ldg(&predictions[row * D + i]);
        float target = __ldg(&targets[row * D + i]);
        sum_dot += pred * target;
        sum_pred_sq += pred * pred;
        sum_target_sq += target * target;
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
    }

    extern __shared__ float shared[];
    int nWarps = (blockSize + warpSize - 1) / warpSize;
    int padded_nWarps = nWarps + 1;
    
    if (lane == 0) {
        shared[warp_id] = sum_dot;
        shared[warp_id + padded_nWarps] = sum_pred_sq;
        shared[warp_id + 2 * padded_nWarps] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction by first warp
    if (tid < warpSize) {
        float final_dot = (tid < nWarps) ? shared[tid] : 0.0f;
        float final_pred_sq = (tid < nWarps) ? shared[tid + nWarps] : 0.0f;
        float final_target_sq = (tid < nWarps) ? shared[tid + 2 * nWarps] : 0.0f;

        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            final_dot += __shfl_down_sync(0xffffffff, final_dot, offset);
            final_pred_sq += __shfl_down_sync(0xffffffff, final_pred_sq, offset);
            final_target_sq += __shfl_down_sync(0xffffffff, final_target_sq, offset);
        }

        if (lane == 0) {
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