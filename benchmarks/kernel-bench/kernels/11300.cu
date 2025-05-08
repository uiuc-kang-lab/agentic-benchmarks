#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void grid_aligned_cosine_similarity_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    const int N,
    const int D) {
    
    // Use 2D grid for better workload distribution
    const int rows_per_block = 4;
    const int tid = threadIdx.x;
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;
    
    // Calculate global row index using 2D grid
    const int row_offset = blockIdx.y * rows_per_block;
    const int row = blockIdx.x * gridDim.x * rows_per_block + row_offset;
    
    // Early exit if row is out of bounds
    if (row >= N) return;
    
    // Calculate starting positions for this thread
    const int items_per_thread = (D + blockDim.x - 1) / blockDim.x;
    const int col_start = tid;
    const int stride = blockDim.x;
    
    // Shared memory for partial results
    __shared__ float s_dot[rows_per_block][32];
    __shared__ float s_pred_sq[rows_per_block][32];
    __shared__ float s_target_sq[rows_per_block][32];
    
    // Process multiple rows per block
    #pragma unroll
    for (int r = 0; r < rows_per_block; r++) {
        const int current_row = row + r;
        if (current_row >= N) break;
        
        const float* pred_row = predictions + current_row * D;
        const float* target_row = targets + current_row * D;
        
        float dot_sum = 0.0f;
        float pred_sq_sum = 0.0f;
        float target_sq_sum = 0.0f;
        
        // Process elements with vectorized loads where possible
        #pragma unroll 4
        for (int i = 0; i < items_per_thread; i++) {
            const int idx = col_start + i * stride;
            if (idx < D) {
                const float pred = pred_row[idx];
                const float target = target_row[idx];
                dot_sum += pred * target;
                pred_sq_sum += pred * pred;
                target_sq_sum += target * target;
            }
        }
        
        // Warp-level reduction
        dot_sum = warp_reduce_sum(dot_sum);
        pred_sq_sum = warp_reduce_sum(pred_sq_sum);
        target_sq_sum = warp_reduce_sum(target_sq_sum);
        
        // Store warp results
        if (lane_id == 0) {
            s_dot[r][warp_id] = dot_sum;
            s_pred_sq[r][warp_id] = pred_sq_sum;
            s_target_sq[r][warp_id] = target_sq_sum;
        }
    }
    __syncthreads();
    
    // Final reduction and output computation
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < rows_per_block; r++) {
            const int current_row = row + r;
            if (current_row >= N) break;
            
            if (lane_id < (blockDim.x / warpSize)) {
                float dot_sum = s_dot[r][lane_id];
                float pred_sq_sum = s_pred_sq[r][lane_id];
                float target_sq_sum = s_target_sq[r][lane_id];
                
                dot_sum = warp_reduce_sum(dot_sum);
                pred_sq_sum = warp_reduce_sum(pred_sq_sum);
                target_sq_sum = warp_reduce_sum(target_sq_sum);
                
                if (lane_id == 0) {
                    const float eps = 1e-8f;
                    float norm_pred = sqrtf(pred_sq_sum);
                    float norm_target = sqrtf(target_sq_sum);
                    float denominator = norm_pred * norm_target;
                    denominator = fmaxf(denominator, eps);
                    float cos_sim = dot_sum / denominator;
                    atomicAdd(output, (1.0f - cos_sim) / N);
                }
            }
        }
    }
}

torch::Tensor grid_aligned_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    
    // Calculate grid dimensions for 2D organization
    const int threads_per_block = 256;
    const int rows_per_block = 4;
    const int blocks_x = 32;
    const int blocks_y = (N + rows_per_block * blocks_x - 1) / (rows_per_block * blocks_x);
    
    dim3 grid(blocks_x, blocks_y);
    
    grid_aligned_cosine_similarity_loss_kernel<<<grid, threads_per_block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &grid_aligned_cosine_similarity_loss_forward, "Grid Aligned Cosine Similarity Loss Forward (CUDA)");
}