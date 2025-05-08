#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void cosine_similarity_loss_kernel_optimized_v2(const float* __restrict__ predictions,
                                                           const float* __restrict__ targets,
                                                           float* output,
                                                           int N,
                                                           int D) {
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int warpSize = 32;
    const int numWarps = blockDim.x / warpSize;
    
    // Align shared memory to avoid bank conflicts
    float* shared_preds = (float*)shared_mem;
    float* shared_targets = shared_preds + ((D + 31) & ~31); // Pad to avoid bank conflicts
    
    // Use float4 for coalesced memory access
    float4* pred_vec = (float4*)predictions;
    float4* target_vec = (float4*)targets;
    float4* shared_pred_vec = (float4*)shared_preds;
    float4* shared_target_vec = (float4*)shared_targets;
    
    // Load data into shared memory using vectorized loads
    const int vec_size = 4;
    const int vec_elements = D / vec_size;
    const int row_offset = row * D;
    
        // Partition vectorized loads among warps for coalesced access
    int numWarps = blockDim.x / warpSize;
    int elements_per_warp = vec_elements / numWarps;
    int remainder = vec_elements % numWarps;
    int start = wid * elements_per_warp + ((wid < remainder) ? wid : remainder);
    int end = start + elements_per_warp + (wid < remainder ? 1 : 0);
    #pragma unroll
    for (int i = start + lane; i < end; i += warpSize) {
        shared_pred_vec[i] = pred_vec[row_offset/vec_size + i];
        shared_target_vec[i] = target_vec[row_offset/vec_size + i];
    }
    
    // Handle remaining elements
    for (int i = tid + vec_elements * vec_size; i < D; i += blockDim.x) {
        shared_preds[i] = predictions[row_offset + i];
        shared_targets[i] = targets[row_offset + i];
    }
    
    __syncthreads();
    
    // Compute partial sums using vectorized operations where possible
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;
    
    #pragma unroll
    for (int i = tid; i < D; i += blockDim.x) {
        float pred = shared_preds[i];
        float target = shared_targets[i];
        sum_dot = __fmaf_rn(pred, target, sum_dot);
        sum_pred_sq = __fmaf_rn(pred, pred, sum_pred_sq);
        sum_target_sq = __fmaf_rn(target, target, sum_target_sq);
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        float* s_dot = shared_mem;
        float* s_pred_sq = s_dot + numWarps;
        float* s_target_sq = s_pred_sq + numWarps;
        
        s_dot[wid] = sum_dot;
        s_pred_sq[wid] = sum_pred_sq;
        s_target_sq[wid] = sum_target_sq;
    }
    
    __syncthreads();
    
    // Final reduction across warps
    if (tid < numWarps) {
        sum_dot = shared_mem[tid];
        sum_pred_sq = shared_mem[numWarps + tid];
        sum_target_sq = shared_mem[2 * numWarps + tid];
        
        #pragma unroll
        for (int offset = numWarps/2; offset > 0; offset >>= 1) {
            sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
            sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
            sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
        }
        
        if (tid == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(sum_pred_sq);
            float norm_target = sqrtf(sum_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = sum_dot / denominator;
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
    
    // Optimize block size for H100
    const int block_size = 256;
    const int numWarps = block_size / 32;
    
    // Align shared memory to avoid bank conflicts
    size_t shared_mem_size = ((D + 31) & ~31) * 2 * sizeof(float) + 3 * numWarps * sizeof(float);
    
    cosine_similarity_loss_kernel_optimized_v2<<<N, block_size, shared_mem_size>>>(
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
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward with optimized shared memory v2 (CUDA)");
}