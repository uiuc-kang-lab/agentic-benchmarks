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

__global__ void uniform_flow_cosine_loss_kernel(const float* __restrict__ predictions,
                                               const float* __restrict__ targets,
                                               float* output,
                                               const int N,
                                               const int D) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;
    
    __shared__ float s_partial_sums[WARPS_PER_BLOCK * 3];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int row = blockIdx.x;
    
    const int base_idx = row * D;
    const float* pred_row = predictions + base_idx;
    const float* target_row = targets + base_idx;
    
    float dot_sum = 0.0f;
    float pred_sq_sum = 0.0f;
    float target_sq_sum = 0.0f;
    
    const int elements_per_thread = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int start_idx = tid;
    
    #pragma unroll 4
    for (int i = 0; i < elements_per_thread; i++) {
        const int idx = start_idx + i * BLOCK_SIZE;
        const bool valid = idx < D;
        
        float pred = valid ? pred_row[idx] : 0.0f;
        float target = valid ? target_row[idx] : 0.0f;
        
        dot_sum += pred * target;
        pred_sq_sum += pred * pred;
        target_sq_sum += target * target;
    }
    
    dot_sum = warp_reduce_sum(dot_sum);
    pred_sq_sum = warp_reduce_sum(pred_sq_sum);
    target_sq_sum = warp_reduce_sum(target_sq_sum);
    
    const bool is_warp_leader = (lane_id == 0);
    if (is_warp_leader) {
        s_partial_sums[warp_id] = dot_sum;
        s_partial_sums[warp_id + WARPS_PER_BLOCK] = pred_sq_sum;
        s_partial_sums[warp_id + 2 * WARPS_PER_BLOCK] = target_sq_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        dot_sum = (lane_id < WARPS_PER_BLOCK) ? s_partial_sums[lane_id] : 0.0f;
        pred_sq_sum = (lane_id < WARPS_PER_BLOCK) ? s_partial_sums[lane_id + WARPS_PER_BLOCK] : 0.0f;
        target_sq_sum = (lane_id < WARPS_PER_BLOCK) ? s_partial_sums[lane_id + 2 * WARPS_PER_BLOCK] : 0.0f;
        
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

torch::Tensor uniform_flow_cosine_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    const int block_size = 256;
    
    uniform_flow_cosine_loss_kernel<<<N, block_size>>>(predictions.data_ptr<float>(),
                                                      targets.data_ptr<float>(),
                                                      output.data_ptr<float>(),
                                                      N,
                                                      D);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &uniform_flow_cosine_loss_forward, "Uniform Flow Cosine Similarity Loss Forward (CUDA)");
}