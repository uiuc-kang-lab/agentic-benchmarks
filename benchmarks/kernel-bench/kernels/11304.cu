#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle instructions
__inline__ __device__ float warp_reduce_sum(float val) {
    // use full mask for all active threads in a warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Combined kernel that processes one row per block
__global__ void optimal_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                         const float* __restrict__ targets,
                                                         float* output,
                                                         int N,
                                                         int D) {
    // Each block processes one row
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    
    // Determine number of warps per block (assuming blockDim.x is a multiple of warpSize or close)
    const int warps_per_block = (blockSize + warpSize - 1) / warpSize;
    
    // Use vectorized loads for coalesced access
    const int vec_size = 4;
    const int D_aligned = (D / vec_size) * vec_size;  // largest multiple of vec_size
    const int numVec = D_aligned / vec_size;

    // Reinterpret pointers to float4 for vectorized loads
    const float4* pred_vec = reinterpret_cast<const float4*>(predictions + row * D);
    const float4* target_vec = reinterpret_cast<const float4*>(targets + row * D);

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Process the vectorized part using coalesced memory accesses
    for (int i = tid; i < numVec; i += blockSize) {
        float4 p = pred_vec[i];
        float4 t = target_vec[i];
        sum_dot      += p.x * t.x + p.y * t.y + p.z * t.z + p.w * t.w;
        sum_pred_sq  += p.x * p.x + p.y * p.y + p.z * p.z + p.w * p.w;
        sum_target_sq+= t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
    }

    // Process remaining tail elements
    for (int i = D_aligned + tid; i < D; i += blockSize) {
        float p = predictions[row * D + i];
        float t = targets[row * D + i];
        sum_dot      += p * t;
        sum_pred_sq  += p * p;
        sum_target_sq+= t * t;
    }

    // Each warp performs its own reduction
    sum_dot = warp_reduce_sum(sum_dot);
    sum_pred_sq = warp_reduce_sum(sum_pred_sq);
    sum_target_sq = warp_reduce_sum(sum_target_sq);

    // Shared memory to hold results from each warp
    __shared__ float s_dot[32];
    __shared__ float s_pred_sq[32];
    __shared__ float s_target_sq[32];

    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    if (lane_id == 0) {
        s_dot[warp_id] = sum_dot;
        s_pred_sq[warp_id] = sum_pred_sq;
        s_target_sq[warp_id] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction across warps performed by the first warp
    if (warp_id == 0 && lane_id < warps_per_block) {
        float dot_val = s_dot[lane_id];
        float pred_sq_val = s_pred_sq[lane_id];
        float target_sq_val = s_target_sq[lane_id];

        dot_val = warp_reduce_sum(dot_val);
        pred_sq_val = warp_reduce_sum(pred_sq_val);
        target_sq_val = warp_reduce_sum(target_sq_val);

        // Only the first thread computes the final value
        if (lane_id == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(pred_sq_val);
            float norm_target = sqrtf(target_sq_val);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = dot_val / denominator;
            float loss = (1.0f - cos_sim) / N;
            // Atomically accumulate the loss across blocks
            atomicAdd(output, loss);
        }
    }
}

// Host function binding the CUDA kernel to PyTorch

torch::Tensor optimal_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());

    // Launch one block per row; using 512 threads per block facilitates full warp occupancy
    const int block_size = 512;
    optimal_cosine_similarity_loss_kernel<<<N, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimal_cosine_similarity_loss_forward, "Optimal Cosine Similarity Loss Forward (CUDA)");
}
