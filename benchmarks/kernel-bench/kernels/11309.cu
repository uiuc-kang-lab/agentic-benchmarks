#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void persistent_cosine_similarity_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    const int N,
    const int D,
    const int rows_per_block) {

    // Shared memory for partial sums
    extern __shared__ float shared_mem[];
    float* block_partial_sums = shared_mem;
    
    const int tid = threadIdx.x;
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;
    
    // Process multiple rows per block to reduce atomic operations
    const int first_row = blockIdx.x * rows_per_block;
    const int last_row = min(first_row + rows_per_block, N);
    
    // Vector load setup
    const int vec_size = 4;
    const int D_aligned = (D / vec_size) * vec_size;
    const int num_vec = D_aligned / vec_size;
    
    for (int row = first_row; row < last_row; row++) {
        const float4* pred_vec = reinterpret_cast<const float4*>(predictions + row * D);
        const float4* target_vec = reinterpret_cast<const float4*>(targets + row * D);
        
        float sum_dot = 0.0f;
        float sum_pred_sq = 0.0f;
        float sum_target_sq = 0.0f;

        // Vectorized loads and computation
        for (int i = tid; i < num_vec; i += blockDim.x) {
            float4 p = pred_vec[i];
            float4 t = target_vec[i];
            
            sum_dot += p.x * t.x + p.y * t.y + p.z * t.z + p.w * t.w;
            sum_pred_sq += p.x * p.x + p.y * p.y + p.z * p.z + p.w * p.w;
            sum_target_sq += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
        }

        // Handle remaining elements
        for (int i = D_aligned + tid; i < D; i += blockDim.x) {
            float p = predictions[row * D + i];
            float t = targets[row * D + i];
            sum_dot += p * t;
            sum_pred_sq += p * p;
            sum_target_sq += t * t;
        }

        // Warp-level reduction
        sum_dot = warp_reduce_sum(sum_dot);
        sum_pred_sq = warp_reduce_sum(sum_pred_sq);
        sum_target_sq = warp_reduce_sum(sum_target_sq);

        // First thread in each warp writes to shared memory
        if (lane_id == 0) {
            const int warp_write_idx = warp_id * 3;
            block_partial_sums[warp_write_idx] = sum_dot;
            block_partial_sums[warp_write_idx + 1] = sum_pred_sq;
            block_partial_sums[warp_write_idx + 2] = sum_target_sq;
        }
        __syncthreads();

        // Final reduction by first warp
        if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
            const int read_idx = lane_id * 3;
            sum_dot = block_partial_sums[read_idx];
            sum_pred_sq = block_partial_sums[read_idx + 1];
            sum_target_sq = block_partial_sums[read_idx + 2];

            sum_dot = warp_reduce_sum(sum_dot);
            sum_pred_sq = warp_reduce_sum(sum_pred_sq);
            sum_target_sq = warp_reduce_sum(sum_target_sq);

            if (lane_id == 0) {
                const float eps = 1e-8f;
                float norm_pred = sqrtf(sum_pred_sq);
                float norm_target = sqrtf(sum_target_sq);
                float denominator = norm_pred * norm_target;
                denominator = fmaxf(denominator, eps);
                float cos_sim = sum_dot / denominator;
                
                // Single atomic operation per block for multiple rows
                block_partial_sums[0] = (1.0f - cos_sim) / N;
            }
        }
        __syncthreads();

        // Only one thread per block performs the final atomic add
        if (tid == 0) {
            atomicAdd(output, block_partial_sums[0]);
        }
        __syncthreads();
    }
}

torch::Tensor persistent_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 512;
    const int rows_per_block = 4;  // Process multiple rows per block
    const int num_blocks = (N + rows_per_block - 1) / rows_per_block;
    
    // Shared memory size: space for partial sums (3 values per warp)
    const int warps_per_block = block_size / warpSize;
    const size_t shared_mem_size = warps_per_block * 3 * sizeof(float);

    persistent_cosine_similarity_loss_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D,
        rows_per_block
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &persistent_cosine_similarity_loss_forward, "Persistent Cosine Similarity Loss Forward (CUDA)");
}